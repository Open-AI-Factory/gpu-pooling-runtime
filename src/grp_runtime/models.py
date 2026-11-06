from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Phase(str, Enum):
    ACCEPTED = "Accepted"
    PREPARING = "Preparing"
    CHECKPOINT = "Checkpoint"
    RECONFIGURE = "Reconfigure"
    SUCCEEDED = "Succeeded"
    ABORTED = "Aborted"

    @property
    def terminal(self) -> bool:
        return self in (Phase.SUCCEEDED, Phase.ABORTED)


class Previous(str, Enum):
    KEPT = "Kept"
    RESTORED = "Restored"
    LOST = "Lost"


ACTIONS = ("Resize", "ScaleUp", "ScaleDown", "Reconfigure", "Restart", "ApplySettings", "Observe")
KEEP_LAYOUT_ACTIONS = ("ApplySettings", "Observe")
RUNTIMES = ("pytorch", "ddp", "fsdp", "pytorch-3d", "deepspeed", "vllm")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class WorkloadRef:
    name: str
    namespace: str = "default"
    kind: str = "Workload"

    @property
    def key(self) -> str:
        return f"{self.namespace}/{self.name}"


@dataclass
class SafePolicy:
    mode: str = "WaitSafePoint"
    timeout_seconds: float = 60


@dataclass
class Profiling:
    workload_type: str = "training"
    model_memory_gb: float = 0.0
    activation_memory_gb: float = 0.0
    num_layers: int | None = None
    num_attention_heads: int | None = None
    weights_gb: float | None = None
    gradients_gb: float | None = None
    optimizer_gb: float | None = None
    activation_gb_per_sample: float | None = None
    global_batch: int | None = None

    @classmethod
    def from_dict(cls, pf: dict[str, Any] | None) -> "Profiling":
        pf = pf or {}
        return cls(
            workload_type=pf.get("workloadType", "training"),
            model_memory_gb=float(pf.get("modelMemoryGb", 0)),
            activation_memory_gb=float(pf.get("activationMemoryGb", 0)),
            num_layers=pf.get("numLayers"),
            num_attention_heads=pf.get("numAttentionHeads"),
            weights_gb=pf.get("weightsGb"),
            gradients_gb=pf.get("gradientsGb"),
            optimizer_gb=pf.get("optimizerGb"),
            activation_gb_per_sample=pf.get("activationGbPerSample"),
            global_batch=pf.get("globalBatch"),
        )

    def model_state_gb(self) -> float:
        parts = [self.weights_gb, self.gradients_gb, self.optimizer_gb]
        if any(p is not None for p in parts):
            return sum(p or 0.0 for p in parts)
        return self.model_memory_gb

    def activation_gb(self, dp: int, pp: int) -> tuple[float, int | None]:
        if self.activation_gb_per_sample is not None and self.global_batch:
            per_rank = self.global_batch // dp
            return per_rank * self.activation_gb_per_sample / pp, per_rank
        return self.activation_memory_gb, None


@dataclass
class RuntimeActionSpec:
    name: str
    workload_ref: WorkloadRef
    runtime: str
    action: str
    target_gpu_count: int
    safe_policy: SafePolicy = field(default_factory=SafePolicy)
    profiling: Profiling = field(default_factory=Profiling)
    adapter_mode: str = "mock"
    adapter_options: dict[str, Any] = field(default_factory=dict)
    pool_snapshot: str | None = None
    mode: str = "restart"
    execution_plan_ref: str | None = None
    namespace: str = "default"
    target_parallelism: dict[str, int] | None = None
    target_gpu_ids: list[str] | None = None
    target_settings: dict[str, Any] | None = None

    @classmethod
    def from_manifest(cls, m: dict[str, Any]) -> "RuntimeActionSpec":
        s = m.get("spec", {})
        md = m.get("metadata", {})
        ns = md.get("namespace", "default")
        wr = s.get("workloadRef", {})
        if isinstance(wr, str):
            wr = {"name": wr, "namespace": ns}
        sp = s.get("safePolicy", {})
        pf = s.get("profiling", {})
        tg = s.get("target", {}) or {}
        return cls(
            name=md.get("name", "runtimeaction"),
            workload_ref=WorkloadRef(
                name=wr.get("name", ""),
                namespace=wr.get("namespace", ns),
                kind=wr.get("kind", "Workload"),
            ),
            runtime=s.get("runtime", ""),
            action=s.get("action", "Resize"),
            target_gpu_count=int((s.get("target", {}) or {}).get("gpus",
                                 (s.get("target", {}) or {}).get("gpuCount", 0))),
            safe_policy=SafePolicy(
                mode=sp.get("mode", "WaitSafePoint"),
                timeout_seconds=float(sp.get("timeoutSeconds", 60)),
            ),
            mode=s.get("mode", "restart"),
            execution_plan_ref=s.get("executionPlanRef"),
            namespace=ns,
            profiling=Profiling.from_dict(pf),
            adapter_mode=s.get("adapterMode", "mock"),
            adapter_options=s.get("adapterOptions", {}) or {},
            pool_snapshot=s.get("poolSnapshot"),
            target_parallelism=tg.get("parallelism"),
            target_gpu_ids=(tg.get("allocation") or {}).get("gpuIds"),
            target_settings=tg.get("settings"),
        )


@dataclass
class ParallelConfig:
    dp: int
    tp: int
    pp: int

    @property
    def gpus(self) -> int:
        return self.dp * self.tp * self.pp

    def __str__(self) -> str:
        return f"DP={self.dp}/TP={self.tp}/PP={self.pp}"


@dataclass
class ExecutionPlan:
    dp: int
    tp: int
    pp: int
    nodes: list[str]
    gpus_per_node: int
    interconnect: str
    adapter: str
    gpu_ids: list[str]
    reason: str
    candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def parallel(self) -> ParallelConfig:
        return ParallelConfig(self.dp, self.tp, self.pp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dp": self.dp, "tp": self.tp, "pp": self.pp,
            "nodes": self.nodes, "gpusPerNode": self.gpus_per_node,
            "interconnect": self.interconnect, "adapter": self.adapter,
            "gpuIds": self.gpu_ids, "reason": self.reason,
            "candidates": self.candidates,
        }


@dataclass
class WorkloadState:
    gpu_ids: list[str]
    parallel: ParallelConfig
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        p = self.parallel
        return {"gpuCount": len(self.gpu_ids), "gpuIds": self.gpu_ids,
                "dp": p.dp, "tp": p.tp, "pp": p.pp, **self.detail}


@dataclass
class RuntimeActionStatus:
    phase: Phase = Phase.ACCEPTED
    mode: str = "restart"
    adapter: str | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    execution_plan: dict[str, Any] | None = None
    safe_point_ready: bool = False
    released_gpus: list[str] = field(default_factory=list)
    reason: str = ""
    result: str = ""
    previous: str | None = None
    message: str = ""
    conditions: list[dict[str, Any]] = field(default_factory=list)
    observation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {
            "phase": self.phase.value, "mode": d["mode"], "adapter": d["adapter"],
            "result": d["result"], "reason": d["reason"], "previous": d["previous"],
            "before": d["before"], "after": d["after"],
            "executionPlan": d["execution_plan"],
            "safePointReady": d["safe_point_ready"],
            "releasedGpus": d["released_gpus"],
            "message": d["message"],
            "conditions": d["conditions"],
            **({"observation": d["observation"]} if self.observation is not None else {}),
        }
