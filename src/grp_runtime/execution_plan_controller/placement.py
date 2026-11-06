from __future__ import annotations

from itertools import permutations
from typing import Any

from ..pool import Gpu, MockPoolClient

FAST_LINKS = ("NVLink", "NVSwitch")
PRIORITY = ("tpCrossNode", "tpOnSlowLink", "dpCrossNode", "ppCrossNode", "nodesUsed", "fragmentedNodes")
RULE = ("rule-based placement comparison (no measured cost): fewer cross-node TP groups, then TP on NVLink/NVSwitch, "
        "then fewer cross-node DP groups, then fewer cross-node PP boundaries, then fewer nodes, "
        "then fewer partially used nodes")


def free_by_node(pool: MockPoolClient, key: str, only: set[str] | None = None) -> dict[str, list[Gpu]]:
    out: dict[str, list[Gpu]] = {}
    for g in sorted(pool.gpus.values(), key=lambda g: (g.node, len(g.id), g.id)):
        if g.allocated_to in (None, key) and (only is None or g.id in only):
            out.setdefault(g.node, []).append(g)
    return out


def gpu_sets(free: dict[str, list[Gpu]], total: int) -> list[dict[str, Any]]:
    out = []
    for gpn in [d for d in range(total, 0, -1) if total % d == 0]:
        need = total // gpn
        ok = [n for n, gs in free.items() if len(gs) >= gpn]
        if len(ok) < need:
            continue
        chosen = sorted(sorted(ok, key=lambda n: (len(free[n]) - gpn, n))[:need])
        out.append({"gpusPerNode": gpn, "nodes": chosen, "gpus": [g for n in chosen for g in free[n][:gpn]],
                    "perNode": {n: gpn for n in chosen},
                    "leftover": {n: len(free[n]) - gpn for n in chosen}})
    return out


def _cell(x: dict[str, int]) -> str:
    return f"D{x['dp'] + 1}P{x['pp'] + 1}T{x['tp'] + 1}"


def _metrics(ranks: list[dict], dp: int, tp: int, pp: int, links: dict[str, str]) -> dict[str, Any]:
    def groups(key):
        g: dict[tuple, set] = {}
        for x in ranks:
            g.setdefault(key(x), set()).add(x["node"])
        return g

    tp_groups = groups(lambda x: (x["dp"], x["pp"]))
    dp_groups = groups(lambda x: (x["tp"], x["pp"]))
    at = {(x["dp"], x["tp"], x["pp"]): x["node"] for x in ranks}
    pp_cross = sum(1 for d in range(dp) for t in range(tp) for p in range(pp - 1)
                   if at[(d, t, p)] != at[(d, t, p + 1)])
    stages: dict[str, set] = {}
    for x in ranks:
        stages.setdefault(x["node"], set()).add(x["pp"])
    tp_cross = sum(1 for v in tp_groups.values() if len(v) > 1) if tp > 1 else 0
    slow = sum(1 for v in tp_groups.values() if len(v) == 1 and links.get(next(iter(v))) not in FAST_LINKS) \
        if tp > 1 else 0
    return {"tpCrossNode": tp_cross, "tpOnSlowLink": slow,
            "dpCrossNode": sum(1 for v in dp_groups.values() if len(v) > 1) if dp > 1 else 0,
            "ppCrossNode": pp_cross,
            "tpGroups": "cross-node" if tp_cross else ("intra-node" if tp > 1 else "-"),
            "ppStages": "one stage per node" if max(len(s) for s in stages.values()) == 1 else "mixed per node",
            "layout": {n: [_cell(x) for x in ranks if x["node"] == n] for n in sorted(stages)}}


def layouts(dp: int, tp: int, pp: int, gs: list[Gpu], links: dict[str, str]) -> list[dict[str, Any]]:
    sizes = {"tp": tp, "dp": dp, "pp": pp}
    coords = [{"rank": r, "tp": r % tp, "dp": (r // tp) % dp, "pp": r // (tp * dp)} for r in range(dp * tp * pp)]
    seen, out = set(), []
    for order in permutations(("tp", "dp", "pp")):
        stride, s = {}, 1
        for d in order:
            stride[d] = s
            s *= sizes[d]
        ranks = []
        for c in coords:
            g = gs[sum(c[d] * stride[d] for d in sizes)]
            ranks.append({**c, "node": g.node, "gpu": g.id, **({"uuid": g.uuid} if g.uuid else {})})
        sig = tuple(x["node"] for x in ranks)
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"fillOrder": list(order), "rankMap": ranks, **_metrics(ranks, dp, tp, pp, links)})
    return out


def candidates(pool: MockPoolClient, key: str, dp: int, tp: int, pp: int,
               only: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    free = free_by_node(pool, key, only)
    total = dp * tp * pp
    if only is not None and sum(map(len, free.values())) == total:
        sets = [{"gpusPerNode": max(map(len, free.values())), "nodes": sorted(free),
                 "gpus": [g for n in sorted(free) for g in free[n]], "leftover": {n: 0 for n in free},
                 "perNode": {n: len(free[n]) for n in sorted(free)}}]
    else:
        sets = gpu_sets(free, total)
    out = []
    for gset in sets:
        frag = sum(1 for v in gset["leftover"].values() if v > 0)
        for lay in layouts(dp, tp, pp, gset["gpus"], pool.interconnect):
            out.append({**lay, "nodes": gset["nodes"], "gpusPerNode": gset["gpusPerNode"], "perNode": gset["perNode"],
                        "gpuIds": [g.id for g in gset["gpus"]], "leftoverFreeGpus": gset["leftover"],
                        "nodesUsed": len(gset["nodes"]), "fragmentedNodes": frag,
                        "memoryGb": min(g.memory_gb for g in gset["gpus"]),
                        "ranksPerPhysical": _share(gset["gpus"])})
    out.sort(key=lambda c: tuple(c[k] for k in PRIORITY))
    for i, c in enumerate(out):
        c["label"] = chr(ord("A") + i) if i < 26 else str(i)
        c["selected"] = i == 0
    return out, {n: len(v) for n, v in free.items()}


def _share(gs: list[Gpu]) -> int:
    n: dict[str, int] = {}
    for g in gs:
        n[g.physical] = n.get(g.physical, 0) + 1
    return max(n.values(), default=1)


def summary(c: dict[str, Any]) -> dict[str, Any]:
    keep = ("label", "selected", "nodes", "gpusPerNode", "tpGroups", "ppStages", "tpCrossNode", "tpOnSlowLink",
            "dpCrossNode", "ppCrossNode", "nodesUsed", "fragmentedNodes", "leftoverFreeGpus", "layout")
    return {k: c[k] for k in keep}
