from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from ..models import ParallelConfig, WorkloadState
from .base import AdapterError
from .local_torch import LocalTorchAdapter

ROOT = Path(__file__).resolve().parents[3]


class TorchrunAdapter(LocalTorchAdapter):
    name = "pytorch-ddp/torchrun"

    def _state(self, spec) -> dict:
        s = super()._state(spec)
        if s:
            s.setdefault("iteration", s.get("step"))
            s.setdefault("dp", s.get("worldSize"))
            s.setdefault("resumedIteration", s.get("resumedStep"))
        return s

    def _spawn(self, spec, dp: int, resume: str | None = None) -> None:
        old = getattr(self, "_proc", None)
        if old is not None and old.poll() is None:
            try:
                old.wait(timeout=30)
            except subprocess.TimeoutExpired:
                old.kill()
        d = self._dir(spec)
        (d / "control.json").unlink(missing_ok=True)
        (d / "state.json").unlink(missing_ok=True)
        opts = spec.adapter_options
        script = Path(opts.get("script", "workloads/train_ddp.py"))
        script = script if script.is_absolute() else ROOT / script
        env = {**os.environ, "GRP_RUN_DIR": str(d), "GRP_CKPT_DIR": str(d / "ckpt" / spec.name),
               "GRP_RESUME": resume or "", "OMP_NUM_THREADS": "1"}
        self._spawn_count += 1
        log = open(d / f"torchrun-{int(time.time())}-{self._spawn_count}.log", "w")
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={dp}", str(script), *map(str, opts.get("scriptArgs", []))]
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        self._log = log.name

    def _ws(self, spec, s: dict, extra: dict | None = None) -> WorkloadState:
        gpus = [g.id for g in self.pool.gpus_of(spec.workload_ref.key)]
        detail = {"runtime": spec.runtime, "mode": "torchrun-ddp", "physicalGpu": s.get("device"),
                  "worldSize": s.get("worldSize"), "rankPids": s.get("rankPids"),
                  "rankMaxMemMb": s.get("rankMaxMemMb"),
                  "physicalGpuMemMb": round(sum(s.get("rankMaxMemMb") or [0]), 1),
                  "step": s.get("step"), "loss": s.get("loss"),
                  "perRankBatch": s.get("perRankBatch"), "globalBatch": s.get("globalBatch"),
                  "note": "one process per rank, DDP gradient all-reduce (gloo, sharing one physical GPU)",
                  **(extra or {})}
        return WorkloadState(gpus, ParallelConfig(s.get("worldSize", 1), 1, 1), detail)

    def reconfigure(self, spec, plan):
        ws = super().reconfigure(spec, plan)
        s = self._state(spec)
        if s.get("worldSize") != plan.dp:
            raise AdapterError(f"world size after restart {s.get('worldSize')} != planned {plan.dp}")
        ws.detail["resumedStep"] = s.get("resumedStep")
        ws.detail["checkpointWorldSize"] = s.get("checkpointWorldSize")
        return ws
