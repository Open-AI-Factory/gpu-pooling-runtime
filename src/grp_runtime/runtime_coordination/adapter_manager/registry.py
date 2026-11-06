from __future__ import annotations

from pathlib import Path

from ...models import RuntimeActionSpec
from ...pool import MockPoolClient
from ...runtime_adapter.base import AdapterError, RuntimeAdapter
from ...runtime_adapter.local_torch import LocalTorchAdapter
from ...runtime_adapter.mock import MockAdapter

ADAPTER_NAMES = {"pytorch": "pytorch-elastic", "ddp": "ddp", "fsdp": "fsdp", "pytorch-3d": "pytorch-3d",
                 "deepspeed": "deepspeed-elastic", "vllm": "vllm-serving"}

REGISTRY: dict[str, dict] = {
    "mock": {"version": "0.1", "runtimes": ["pytorch", "ddp", "fsdp", "deepspeed", "vllm"],
             "reconfigure": ["restart"], "tp": True, "pp": True, "platform": "none",
             "note": "state transitions only, no execution"},
    "local": {"version": "0.1", "runtimes": ["pytorch"], "reconfigure": ["restart"],
              "tp": False, "pp": False, "platform": "host (single process, DP emulation)"},
    "torchrun": {"version": "0.2", "runtimes": ["pytorch", "ddp", "fsdp"], "reconfigure": ["restart"],
                 "tp": False, "pp": False, "platform": "host Linux (torchrun multi-process)"},
    "k8s": {"version": "0.3", "runtimes": ["ddp", "fsdp", "pytorch-3d"], "reconfigure": ["restart"],
            "tp": False, "pp": False, "tpPpRuntimes": ["pytorch-3d"], "reshardRuntimes": ["pytorch-3d"],
            "platform": "Kubernetes (rank Pod, Agent ConfigMap)",
            "note": "pytorch-3d launches DP×TP×PP and reshards its layout-free checkpoint on reconfigure"},
}


def capability_problems(mode: str, runtime: str, tp: int = 1, pp: int = 1,
                        reconfigure: str = "restart") -> list[str]:
    cap = REGISTRY.get(mode)
    if cap is None:
        return [f"unknown adapterMode: {mode}"]
    out = []
    if runtime not in cap["runtimes"]:
        out.append(f"{mode} Adapter does not support runtime '{runtime}' (supported: {cap['runtimes']})")
    split_ok = runtime in cap.get("tpPpRuntimes", [])
    if tp > 1 and not (cap["tp"] or split_ok):
        out.append(f"{mode} Adapter does not support TP>1")
    if pp > 1 and not (cap["pp"] or split_ok):
        out.append(f"{mode} Adapter does not support PP>1")
    if reconfigure not in cap["reconfigure"]:
        out.append(f"{mode} Adapter does not support reconfigure mode '{reconfigure}'")
    return out


def select_adapter(spec: RuntimeActionSpec, pool: MockPoolClient, runs_root: Path) -> RuntimeAdapter:
    problems = capability_problems(spec.adapter_mode, spec.runtime, reconfigure=spec.mode)
    if problems:
        raise AdapterError("; ".join(problems))
    if spec.adapter_mode == "mock":
        return MockAdapter(f"{ADAPTER_NAMES[spec.runtime]}/mock", pool, spec.pool_snapshot)
    if spec.adapter_mode == "local":
        return LocalTorchAdapter(pool, runs_root)
    if spec.adapter_mode == "torchrun":
        from ...runtime_adapter.torchrun import TorchrunAdapter
        return TorchrunAdapter(pool, runs_root)
    from ...runtime_adapter.k8s import K8sAdapter
    return K8sAdapter(spec, pool)


__all__ = ["AdapterError", "RuntimeAdapter", "select_adapter", "ADAPTER_NAMES", "REGISTRY",
           "capability_problems"]
