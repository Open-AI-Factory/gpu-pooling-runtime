from __future__ import annotations

from typing import Any

from ..runtime_coordination.adapter_manager.registry import ADAPTER_NAMES, REGISTRY, capability_problems
from ..models import Profiling
from . import placement, profiles
from .planner import evaluate
from ..pool import Candidate, MockPoolClient

PLACED_BY = "scheduler-mock"
DIST_PORT = 29500
DRA_GPU_CLASS, DRA_MIG_CLASS, DRA_ATTR = "gpu.nvidia.com", "mig.nvidia.com", "gpu.nvidia.com"


def uses_gpu(launch: dict[str, Any] | None) -> bool:
    return int((launch or {}).get("gpu", 0)) > 0


def device_binding(rank_map: list[dict], launch: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if not uses_gpu(launch):
        return {"mode": "none", "note": "ranks run without GPU"}, []
    missing = [f"rank{x['rank']}={x['gpu']}" for x in rank_map if not x.get("uuid")]
    if missing:
        return {"mode": "dra", "claims": []}, missing
    claims = [{"rank": x["rank"], "node": x["node"], "uuid": x["uuid"],
               "deviceClassName": DRA_MIG_CLASS if x["uuid"].startswith("MIG-") else DRA_GPU_CLASS}
              for x in sorted(rank_map, key=lambda x: x["rank"])]
    return {"mode": "dra", "claims": claims}, []


def claim_manifest(name: str, claim: dict[str, Any], labels: dict[str, str], owners: list[dict] | None) -> dict:
    return {"apiVersion": "resource.k8s.io/v1", "kind": "ResourceClaim",
            "metadata": {"name": name, "labels": labels,
                         "annotations": {"grp.keti.re.kr/device-uuid": claim["uuid"]},
                         **({"ownerReferences": owners} if owners else {})},
            "spec": {"devices": {"requests": [{"name": "gpu", "exactly": {
                "deviceClassName": claim["deviceClassName"], "allocationMode": "ExactCount", "count": 1,
                "selectors": [{"cel": {"expression": f'device.attributes["{DRA_ATTR}"].uuid == "{claim["uuid"]}"'}}],
            }}]}}}


def process_groups(rank_map: list[dict], dp: int, tp: int, pp: int) -> dict[str, list[list[int]]]:
    at = {(x["dp"], x["tp"], x["pp"]): x["rank"] for x in rank_map}
    return {"tp": [[at[(d, t, p)] for t in range(tp)] for p in range(pp) for d in range(dp)],
            "dp": [[at[(d, t, p)] for d in range(dp)] for p in range(pp) for t in range(tp)],
            "pp": [[at[(d, t, p)] for p in range(pp)] for d in range(dp) for t in range(tp)]}


def communication(pool: MockPoolClient, nodes: list[str], launch: dict[str, Any] | None) -> dict[str, Any]:
    env = (launch or {}).get("env") or {}
    links = {pool.interconnect.get(n, "unknown") for n in nodes}
    if env.get("GRP_DIST_BACKEND"):
        backend, why = env["GRP_DIST_BACKEND"], "set by request launch.env"
    elif launch is not None and not int(launch.get("gpu", 0)):
        backend, why = "gloo", "ranks run without GPU"
    elif "MIG" in links:
        backend, why = "gloo", "MIG instances: NCCL cannot connect peers across MIG"
    else:
        backend, why = "nccl", "GPU ranks without MIG"
    return {"backend": backend, "reason": why, "masterRank": 0, "port": DIST_PORT}


def execution_settings(pool: MockPoolClient, rank_map: list[dict], launch: dict[str, Any] | None,
                       par: dict[str, int], namespace: str | None = None) -> tuple[dict[str, Any] | None, str]:
    devices = [pool.gpus[x["gpu"]] for x in rank_map if x.get("gpu") in pool.gpus]
    return profiles.select(launch, par, devices, namespace)


def rank_plan(pool: MockPoolClient, key: str, dp: int, tp: int, pp: int, gpu_ids: list[str],
              launch: dict[str, Any] | None, namespace: str | None = None) -> tuple[dict[str, Any] | None, str]:
    cands, free = placement.candidates(pool, key, dp, tp, pp, set(gpu_ids))
    if not cands:
        return None, f"GPUs {gpu_ids} cannot host DP{dp} TP{tp} PP{pp} (usable per node {free or '{}'})"
    best = cands[0]
    binding, missing = device_binding(best["rankMap"], launch)
    if missing:
        return None, f"DRA needs a device UUID per rank; pool {pool.version} has none for {missing}"
    settings, why = execution_settings(pool, best["rankMap"], launch, {"dp": dp, "tp": tp, "pp": pp}, namespace)
    if why:
        return None, why
    return {"selected": {"dp": dp, "tp": tp, "pp": pp}, "rankMap": best["rankMap"], "settings": settings,
            "processGroups": process_groups(best["rankMap"], dp, tp, pp),
            "communication": communication(pool, best["nodes"], launch), "deviceBinding": binding,
            "gpusPerNode": best["gpusPerNode"], "perNode": best["perNode"]}, ""


def planned_ranks(spec: dict[str, Any], dp: int, tp: int, pp: int, nodes: list[str]) -> list[dict] | None:
    planning = spec.get("planning") or {}
    rmap = sorted(planning.get("rankMap") or [], key=lambda x: x["rank"])
    sel = planning.get("selected") or {}
    if len(rmap) != dp * tp * pp or (sel.get("dp"), sel.get("tp"), sel.get("pp")) != (dp, tp, pp):
        return None
    if not {x["node"] for x in rmap} <= set(nodes):
        return None
    return rmap


def select_nodes(pool: MockPoolClient, nodes: int, gpus_per_node: int, workload_key: str) -> list[str]:
    by_node: dict[str, int] = {}
    mine: set[str] = set()
    for g in pool.gpus.values():
        if g.allocated_to in (None, workload_key):
            by_node[g.node] = by_node.get(g.node, 0) + 1
        if g.allocated_to == workload_key:
            mine.add(g.node)
    ok = sorted((n for n, c in by_node.items() if c >= gpus_per_node), key=lambda n: (n not in mine, n))
    return ok[:nodes]


def pool_topology(pool: MockPoolClient, gpus_per_node: int, workload_key: str,
                  only_nodes: list[str] | None = None) -> dict[str, Any]:
    usable = [g for g in pool.gpus.values() if g.allocated_to in (None, workload_key)
              and (only_nodes is None or g.node in only_nodes)]
    per_node: dict[str, int] = {}
    for g in usable:
        per_node[g.node] = per_node.get(g.node, 0) + 1
    eligible = sorted(n for n, c in per_node.items() if c >= gpus_per_node)
    gs = [g for g in usable if g.node in eligible]
    links = sorted({pool.interconnect.get(n, "unknown") for n in eligible})
    share: dict[str, int] = {}
    for g in gs:
        share[g.physical] = share.get(g.physical, 0) + 1
    return {"usableGpusPerNode": per_node, "eligibleNodes": eligible,
            "memoryGb": min((g.memory_gb for g in gs), default=None),
            "interconnect": links[0] if len(links) == 1 else ("mixed" if links else None),
            "ranksPerPhysical": max(share.values(), default=1)}


def _spec(ep: dict[str, Any]):
    spec = ep.get("spec", {})
    par = spec.get("parallelism", {}) or {}
    pl = spec.get("placement", {}) or {}
    key = f"{ep['metadata'].get('namespace', 'default')}/{spec.get('workloadRef', '')}"
    return (spec, int(par.get("dp", 1)), int(par.get("tp", 1)), int(par.get("pp", 1)),
            int(pl.get("nodes", 1)), int(pl.get("gpusPerNode", 1)), key)


def per_node(ep: dict[str, Any]) -> dict[str, int] | None:
    return ((ep.get("spec") or {}).get("placement") or {}).get("perNode") or None


def check_plan(ep: dict[str, Any], pool: MockPoolClient) -> tuple[list[str], dict | None]:
    spec, dp, tp, pp, nodes, gpn, key = _spec(ep)
    runtime = spec.get("runtime", "")
    plan_only = bool((spec.get("planning") or {}).get("planOnly"))
    problems = [] if plan_only else capability_problems("k8s", runtime, tp, pp)
    uneven = per_node(ep)
    if uneven:
        if dp * tp * pp != sum(uneven.values()) or len(uneven) != nodes:
            problems.append(f"DP×TP×PP={dp * tp * pp} ≠ GPUs per node {uneven}")
    elif dp * tp * pp != nodes * gpn:
        problems.append(f"DP×TP×PP={dp * tp * pp} ≠ nodes×gpusPerNode={nodes * gpn}")
    if tp > gpn and not plan_only and not uneven:
        problems.append(f"TP={tp} > gpusPerNode={gpn} (TP requires intra-node communication)")
    if (spec.get("launch") or {}).get("deviceBinding") == "dra" and \
            len(((spec.get("planning") or {}).get("deviceBinding") or {}).get("claims") or []) != dp * tp * pp:
        problems.append("deviceBinding=dra requires one device claim per rank in the plan")
    topo = pool_topology(pool, min(uneven.values()) if uneven else gpn, key)
    estimate = None
    if topo["memoryGb"] is not None:
        cand = Candidate([], [f"node{i}" for i in range(nodes)], spec.get("interconnect", "Network"), gpn,
                         topo["memoryGb"], ranks_per_physical=topo["ranksPerPhysical"])
        ok, why, per_rank = evaluate(dp, tp, pp, Profiling.from_dict(spec.get("profiling")), cand,
                                     shard_dp=runtime == "fsdp")
        estimate = {"perRankGb": round(per_rank, 3), "note": why}
        if not ok:
            problems.append(f"insufficient memory: {why}")
    return problems, estimate


def place(ep: dict[str, Any], pool: MockPoolClient) -> dict[str, Any]:
    spec, dp, tp, pp, nodes, gpn, key = _spec(ep)
    world = dp * tp * pp
    pinned = (spec.get("planning") or {}).get("pinnedNodes")
    launched = bool((ep.get("status") or {}).get("generation"))
    need = per_node(ep) or {}
    if pinned and not launched:
        cap = pool_topology(pool, gpn, key, only_nodes=list(pinned))["usableGpusPerNode"]
        short = {n: cap.get(n, 0) for n in pinned if cap.get(n, 0) < need.get(n, gpn)}
        chosen = list(pinned) if not short and len(pinned) == nodes else []
        reason = (f"pinned node(s) lack {need or gpn} GPU(s) per node in pool {pool.version}: {short}" if short else
                  f"pinnedNodes {list(pinned)} ≠ placement.nodes={nodes}" if len(pinned) != nodes else
                  "placed on node(s) fixed in the plan")
    else:
        chosen = select_nodes(pool, nodes, gpn, key)
        reason = (f"{nodes} node(s) x {gpn} GPU selected from pool {pool.version} (workload's nodes first, then by name)"
                  if len(chosen) == nodes else
                  f"cannot secure {nodes} node(s) with gpusPerNode={gpn} from resource pool (snapshot {pool.version}) "
                  f"(available {chosen}, usable GPUs per node "
                  f"{pool_topology(pool, gpn, key)['usableGpusPerNode']})")
    if len(chosen) != nodes:
        return {"by": PLACED_BY, "state": "Unschedulable", "nodes": [], "rankNodes": [], "reason": reason}
    rmap = planned_ranks(spec, dp, tp, pp, chosen) if not launched else None
    if rmap:
        rank_nodes, source = [x["node"] for x in rmap], "plan"
    else:
        rank_nodes, source = [chosen[r // gpn] for r in range(world)], "sequential"
    return {"by": PLACED_BY, "state": "Placed", "nodes": chosen, "rankNodes": rank_nodes, "rankSource": source,
            "reason": reason}


def validate(ep: dict[str, Any], pool: MockPoolClient, place_workers: bool = True) -> dict[str, Any]:
    spec = ep.get("spec", {})
    runtime = spec.get("runtime", "")
    problems, estimate = check_plan(ep, pool)
    out = {
        "validation": "Rejected" if problems else "Accepted",
        "reason": "; ".join(problems) if problems else (
            "configuration and memory checks passed (plan only: "
            f"{((spec.get('planning') or {}).get('execution') or {}).get('reason', 'not executable')})"
            if (spec.get("planning") or {}).get("planOnly") else "runtime, configuration and memory checks passed"),
        "adapter": ADAPTER_NAMES.get(runtime, runtime),
        "adapterCapability": {"mode": "k8s", **REGISTRY["k8s"]},
        "memoryEstimate": estimate,
    }
    if place_workers:
        pl = place(ep, pool) if not problems else None
        out["placement"] = pl
        out["assignedNodes"] = pl["nodes"] if pl and pl["state"] == "Placed" else []
    return out
