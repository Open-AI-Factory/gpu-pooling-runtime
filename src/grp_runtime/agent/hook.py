from __future__ import annotations

import hashlib
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .channel import FileChannel, from_env

REQUESTS = {"none": 0, "safepoint": 1, "stop": 2}


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _strategy(model) -> str:
    if hasattr(model, "module"):
        return "ddp"
    if "FSDP" in type(model).__name__:
        return "fsdp"
    return "single"


def _cgroup_isolation() -> dict:
    out: dict = {"cpuLimit": None, "memoryLimitMb": None}
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        out["cpuLimit"] = None if quota == "max" else round(int(quota) / int(period), 2)
    except (OSError, ValueError):
        pass
    try:
        m = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        out["memoryLimitMb"] = None if m == "max" else int(m) // 2**20
    except (OSError, ValueError):
        pass
    return out


def _peak_rss_mb() -> float | None:
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except (ImportError, OSError):
        return None


def _file_digest(path: Path) -> str:
    h = hashlib.sha256(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(2**20))
    return h.hexdigest()[:16]


class RuntimeHook:
    def __init__(self, report_every: int = 10, device: torch.device | None = None):
        self.channel = from_env()
        default_ckpt = os.environ.get("GRP_RUN_DIR", ".")
        self.ckpt_dir = Path(os.environ.get("GRP_CKPT_DIR", default_ckpt))
        self.resume = os.environ.get("GRP_RESUME") or None
        self.ckpt_keep = int(os.environ.get("GRP_CKPT_KEEP", 3))
        self.safepoint_at = int(os.environ.get("GRP_SAFEPOINT_AT_STEP", 0))
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world = dist.get_world_size() if self.distributed else 1
        self.report_every = report_every
        self.device = device
        self._remote = not isinstance(self.channel, FileChannel)
        self._min_check = 0.5 if self._remote else 0.0
        self._min_report = 1.0 if self._remote else 0.0
        self._last_check = self._last_report = 0.0
        self._t0 = time.perf_counter()
        self._start_step = 0
        self.info = {"worldSize": self.world, "resumedFrom": self.resume}
        if self.rank == 0:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._register("connected")

    def _register(self, hook_state: str, step: int | None = None) -> None:
        dev = self.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.channel.write(f"agent-{self.rank}", {
            "hook": hook_state, "rank": self.rank, "world": self.world, "pid": os.getpid(),
            "node": os.environ.get("NODE_NAME", socket.gethostname()), "pod": os.environ.get("POD_NAME"),
            "step": step, "updatedAt": time.time(),
            "isolation": {**_cgroup_isolation(), "device": dev.type,
                          "visibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "threads": torch.get_num_threads()},
        })

    def _load_everywhere(self) -> dict:
        if not self.distributed:
            return torch.load(self.resume, map_location="cpu")
        t = torch.tensor([0 if Path(self.resume).is_file() else 1], dtype=torch.int32)
        if dist.get_backend() == "nccl":
            t = t.cuda()
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if not int(t.item()):
            return torch.load(self.resume, map_location="cpu", mmap=True)
        obj = [torch.load(self.resume, map_location="cpu") if self.rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def restore(self, model, optimizer) -> int:
        if not self.resume:
            return 0
        load_full = getattr(_unwrap(model), "load_full_checkpoint", None)
        if load_full is not None:
            ck = self._load_everywhere()
            load_full(ck, optimizer)
        elif not self.distributed or _strategy(model) == "single" and self.world == 1:
            ck = torch.load(self.resume, map_location="cpu")
            _unwrap(model).load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
        else:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict
            obj = [torch.load(self.resume, map_location="cpu") if self.rank == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            ck = obj[0]
            set_state_dict(model, optimizer, model_state_dict=ck["model"], optim_state_dict=ck["optimizer"],
                           options=StateDictOptions(full_state_dict=True))
        self._start_step = ck["step"]
        self.info.update(resumedStep=ck["step"], lossAtCheckpoint=ck.get("loss"),
                         checkpointWorldSize=ck.get("worldSize"), checkpointStrategy=ck.get("strategy"))
        return ck["step"]

    def step_end(self, step: int, model, optimizer, **metrics) -> bool:
        req = self._agree_request(step)
        if req == REQUESTS["safepoint"]:
            self._checkpoint(step, model, optimizer, metrics)
            return True
        if req == REQUESTS["stop"]:
            self._report("stopped", step, metrics, force=True)
            self._register("disconnected", step)
            return True
        if step % self.report_every == 0:
            self._report("running", step, metrics)
        return False

    def finish(self, step: int, **metrics) -> None:
        self._report("finished", step, metrics, force=True)
        self._register("disconnected", step)

    def _agree_request(self, step: int) -> int:
        local = 0
        if self.rank == 0:
            nowt = time.monotonic()
            if self.safepoint_at and step == self.safepoint_at:
                local = REQUESTS["safepoint"]
            elif nowt - self._last_check >= self._min_check:
                self._last_check = nowt
                ctl = self.channel.read("control") or {}
                local = REQUESTS.get(ctl.get("request", "none"), 0)
        if not self.distributed:
            return local
        t = torch.tensor([local], dtype=torch.int32)
        if dist.get_backend() == "nccl":
            t = t.cuda()
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    def _checkpoint(self, step, model, optimizer, metrics) -> None:
        path = self.ckpt_dir / f"ckpt-step{step}.pt"
        strategy = _strategy(model)
        full = getattr(_unwrap(model), "full_checkpoint", None)
        if full is not None:
            msd, osd = full(optimizer) or (None, None)
            strategy = "layout-free"
        elif self.distributed:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
            msd, osd = get_state_dict(model, optimizer,
                                      options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        else:
            msd, osd = _unwrap(model).state_dict(), optimizer.state_dict()
        ckinfo = {}
        if self.rank == 0:
            torch.save({"model": msd, "optimizer": osd, "step": step, "worldSize": self.world,
                        "strategy": strategy, **metrics}, path)
            ckinfo = {"checkpoint": str(path), "checkpointBytes": path.stat().st_size,
                      "checkpointDigest": _file_digest(path), "strategy": strategy,
                      "checkpointsPruned": self._prune(path)}
        if self.distributed:
            dist.barrier()
        if self.rank == 0:
            self.channel.delete("control")
        self._report("safepoint", step, {**metrics, **ckinfo}, force=True)
        self._register("disconnected", step)

    def _prune(self, latest: Path) -> list[str]:
        if self.ckpt_keep <= 0:
            return []
        protected = {latest.resolve()}
        if self.resume:
            protected.add(Path(self.resume).resolve())
        files = sorted(self.ckpt_dir.glob("ckpt-step*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        removed = []
        for p in files[self.ckpt_keep:]:
            if p.resolve() not in protected:
                p.unlink(missing_ok=True)
                removed.append(p.name)
        return removed

    def _report(self, status: str, step: int, metrics: dict, force: bool = False) -> None:
        nowt = time.monotonic()
        due = force or nowt - self._last_report >= self._min_report
        if self.distributed:
            t = torch.tensor([1 if due else 0], dtype=torch.int32)
            if dist.get_backend() == "nccl":
                t = t.cuda()
            dist.broadcast(t, src=0)
            due = bool(t.item())
        if not due:
            return
        self._last_report = nowt
        mem = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0
        row = (os.getpid(), mem, _peak_rss_mb())
        rows = [row]
        if self.distributed:
            rows = [None] * self.world
            dist.all_gather_object(rows, row)
        if self.rank == 0:
            el = time.perf_counter() - self._t0
            self.channel.write("state", {
                **self.info, "status": status, "step": step,
                "rankPids": [r[0] for r in rows], "rankMaxMemMb": [r[1] for r in rows],
                "rankPeakRssMb": [r[2] for r in rows],
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "stepsPerSec": round((step - self._start_step) / el, 1) if el else None,
                "updatedAt": time.time(), **metrics})
        if status == "running" and self._remote:
            self._register("connected", step)
