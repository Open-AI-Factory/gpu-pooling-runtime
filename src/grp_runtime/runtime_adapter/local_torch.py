from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from ..models import ExecutionPlan, ParallelConfig, RuntimeActionSpec, WorkloadState
from ..pool import MockPoolClient
from .base import AdapterError, RuntimeAdapter

WORKER = Path(__file__).with_name("torch_worker.py")


class LocalTorchAdapter(RuntimeAdapter):
    name = "pytorch-elastic/local-gpu"

    def __init__(self, pool: MockPoolClient, runs_root: Path):
        self.pool = pool
        self.runs_root = runs_root
        self._spawn_count = 0

    def _dir(self, spec) -> Path:
        d = self.runs_root / "local-workloads" / f"{spec.workload_ref.namespace}_{spec.workload_ref.name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _state(self, spec) -> dict:
        p = self._dir(spec) / "state.json"
        for _ in range(20):
            try:
                return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            except (json.JSONDecodeError, OSError):
                time.sleep(0.02)
        return {}

    def _spawn(self, spec, dp: int, resume: str | None = None) -> None:
        d = self._dir(spec)
        (d / "control.json").unlink(missing_ok=True)
        (d / "state.json").unlink(missing_ok=True)
        self._spawn_count += 1
        log = open(d / f"worker-{int(time.time())}-{self._spawn_count}.log", "w")
        cmd = [sys.executable, str(WORKER), "--run-dir", str(d), "--dp", str(dp)]
        if resume:
            cmd += ["--resume", resume]
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        self._log = log.name

    def _wait(self, spec, cond, timeout: float, what: str) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            s = self._state(spec)
            if s and cond(s):
                return s
            proc = getattr(self, "_proc", None)
            if proc is not None and proc.poll() not in (None, 0):
                tail = Path(self._log).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
                raise AdapterError(f"worker exited abnormally during {what} (exit={proc.returncode}): {' '.join(tail)}")
            time.sleep(0.1)
        raise AdapterError(f"timed out waiting for {what} ({timeout:.0f}s), last state={self._state(spec)}")

    def _ws(self, spec, s: dict, extra: dict | None = None) -> WorkloadState:
        gpus = [g.id for g in self.pool.gpus_of(spec.workload_ref.key)]
        detail = {"runtime": spec.runtime, "mode": "local-gpu", "physicalGpu": s.get("device"),
                  "emulation": "DP ranks emulated with gradient accumulation on one physical GPU",
                  "pid": s.get("pid"), "iteration": s.get("iteration"),
                  "loss": s.get("loss"), "microBatch": s.get("microBatch"), **(extra or {})}
        return WorkloadState(gpus, ParallelConfig(s.get("dp", 1), 1, 1), detail)

    def current_state(self, spec):
        s = self._state(spec)
        if s.get("status") != "running":
            dp = int(spec.adapter_options.get("initialDp", 2))
            key = spec.workload_ref.key
            if not self.pool.gpus_of(key):
                free = self.pool.free_gpus()
                if len(free) < dp:
                    raise AdapterError(f"not enough slots for initial launch (need {dp})")
                self.pool.assign(key, [g.id for g in free[:dp]])
            self._spawn(spec, dp)
        warm = int(spec.adapter_options.get("warmupIters", 200))
        s = self._wait(spec, lambda s: s.get("status") == "running" and s["iteration"] >= warm,
                       120, "workload start")
        return self._ws(spec, s)

    def prepare(self, spec, plan: ExecutionPlan):
        if plan.tp > 1 or plan.pp > 1:
            raise AdapterError(f"{plan.parallel}: TP/PP>1 requires multiple physical GPUs (only 1 available)")

    def request_safe_point(self, spec):
        (self._dir(spec) / "control.json").write_text(json.dumps({"request": "safepoint"}), encoding="utf-8")

    def safe_point_ready(self, spec):
        return self._state(spec).get("status") == "safepoint"

    def cancel_safe_point(self, spec):
        (self._dir(spec) / "control.json").unlink(missing_ok=True)
        s = self._state(spec)
        if s.get("status") == "safepoint":
            self._spawn(spec, s["dp"], s["checkpoint"])

    def reconfigure(self, spec, plan):
        sp = self._state(spec)
        ckpt = sp["checkpoint"]
        if spec.adapter_options.get("failAt") == "Reconfigure":
            ckpt = ckpt + ".missing"
        self._sp_info = sp
        self._spawn(spec, plan.dp, ckpt)
        target = sp["iteration"] + 50
        try:
            s = self._wait(spec, lambda s: s.get("status") == "running" and s["iteration"] >= target,
                           60, "training resume after reconfigure")
        except AdapterError as e:
            raise AdapterError(f"reconfigure failed: {e}") from e
        ws = self._ws(spec, s, {"resumedIteration": s.get("resumedIteration"),
                                "lossAtSafePoint": sp["loss"], "checkpoint": ckpt})
        ws.gpu_ids = plan.gpu_ids
        return ws

    def rollback(self, spec, before):
        sp = getattr(self, "_sp_info", None) or self._state(spec)
        if sp.get("checkpoint"):
            self._spawn(spec, before.parallel.dp, sp["checkpoint"])
            self._wait(spec, lambda s: s.get("status") == "running", 60, "resume after rollback")

    def finalize(self, spec):
        if not spec.adapter_options.get("keepRunning", False):
            (self._dir(spec) / "control.json").write_text(json.dumps({"request": "stop"}), encoding="utf-8")
            try:
                self._wait(spec, lambda s: s.get("status") in ("stopped", "finished"), 15, "worker stop")
            except AdapterError:
                pass
