from __future__ import annotations

from typing import Callable

from ..models import ExecutionPlan, Profiling
from ..pool import Candidate

MEMORY_HEADROOM = 0.9
SELECTION_RULE = ("rule-based (no cost/throughput comparison): among feasible combinations, smallest model split "
                  "TP×PP first (max DP); if a split is needed, TP on a single NVLink/NVSwitch node, otherwise PP")
FAST_LINKS = ("NVLink", "NVSwitch")


def enumerate_configs(gpu_count: int) -> list[tuple[int, int, int]]:
    out = []
    for tp in range(1, gpu_count + 1):
        if gpu_count % tp:
            continue
        for pp in range(1, gpu_count // tp + 1):
            if (gpu_count // tp) % pp:
                continue
            out.append((gpu_count // (tp * pp), tp, pp))
    return out


def evaluate(dp: int, tp: int, pp: int, prof: Profiling, cand: Candidate,
             shard_dp: bool = False) -> tuple[bool, str, float]:
    model = prof.model_state_gb() / (tp * pp * (dp if shard_dp else 1))
    act, per_rank_batch = prof.activation_gb(dp, pp)
    mem = model + act
    share = cand.ranks_per_physical
    need = mem * share
    limit = cand.memory_gb * MEMORY_HEADROOM
    detail = f"model{'(FSDP sharded)' if shard_dp else ''} {model:.2f} + activation {act:.2f}"
    if per_rank_batch is not None:
        detail += f"(per-rank batch {per_rank_batch})"
    detail += f" = {mem:.2f}GB per rank"
    if share > 1:
        detail += f" x {share} ranks sharing a physical GPU = {need:.2f}GB"
    if tp > cand.gpus_per_node:
        return False, f"TP={tp} exceeds GPUs per node ({cand.gpus_per_node}) (TP requires intra-node communication)", mem
    if prof.num_attention_heads and prof.num_attention_heads % tp:
        return False, f"{prof.num_attention_heads} attention heads not divisible by TP={tp}", mem
    if prof.num_layers and pp > 1 and prof.num_layers % pp:
        return False, f"{prof.num_layers} layers not divisible by PP={pp}", mem
    if need > limit:
        return False, f"{detail} > limit {limit:.1f}GB", mem
    return True, f"{detail} <= limit {limit:.1f}GB", mem


def build_plan(gpu_count: int, prof: Profiling, cand: Candidate, adapter: str,
               shard_dp: bool = False,
               exclude: Callable[[int, int, int], str | None] | None = None) -> ExecutionPlan | None:
    fast = cand.interconnect in FAST_LINKS and len(cand.nodes) == 1
    rows, feasible = [], []
    for dp, tp, pp in enumerate_configs(gpu_count):
        ok, why, mem = evaluate(dp, tp, pp, prof, cand, shard_dp)
        rule = exclude(dp, tp, pp) if exclude else None
        if ok and rule:
            ok, why = False, rule
        row = {"dp": dp, "tp": tp, "pp": pp, "memPerGpuGb": round(mem, 2),
               "feasible": ok, "note": why}
        rows.append(row)
        if ok:
            feasible.append(((tp * pp, pp if fast else tp), row))
    if not feasible:
        return None
    feasible.sort(key=lambda x: x[0])
    best = feasible[0][1]
    best["selected"] = True
    split = best["tp"] * best["pp"]
    if split == 1:
        reason = f"model fits on one GPU, so all GPUs are used for data parallelism with DP={best['dp']}"
    else:
        pref = "TP" if fast else "PP"
        reason = (f"model must be split {split}-way to fit memory; {cand.interconnect}"
                  f"{' (single node)' if len(cand.nodes) == 1 else ' (inter-node)'} link, so {pref} is preferred")
    return ExecutionPlan(
        dp=best["dp"], tp=best["tp"], pp=best["pp"],
        nodes=cand.nodes, gpus_per_node=cand.gpus_per_node,
        interconnect=cand.interconnect, adapter=adapter,
        gpu_ids=cand.gpu_ids, reason=reason, candidates=rows,
    )
