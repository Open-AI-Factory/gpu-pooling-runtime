from __future__ import annotations

import json
from typing import Any

from ..pool import POOL_GROUP, POOL_PLURAL, POOL_VERSION, _holder

GROUP, VERSION = "grp.keti.re.kr", "v1alpha1"
LABEL = "grp.keti.re.kr/workload"
PROGRESS_KEYS = ("status", "step", "loss", "stepsPerSec", "worldSize", "resumedStep", "resumedFrom",
                 "appliedSettings", "globalBatch", "device", "mlflowRunId", "updatedAt")
ACTION_KEYS = ("phase", "result", "reason", "previous", "message", "before", "after", "conditions", "observation")


def fetch(namespace: str, pool: str) -> dict[str, Any]:
    from kubernetes import client
    co, core = client.CustomObjectsApi(), client.CoreV1Api()
    raw: dict[str, Any] = {kind: co.list_namespaced_custom_object(GROUP, VERSION, namespace, kind)["items"]
                           for kind in ("planrequests", "executionplans", "runtimeactions")}
    raw["progress"] = {}
    for cm in core.list_namespaced_config_map(namespace, label_selector=LABEL).items:
        state = (cm.data or {}).get("state")
        if state:
            raw["progress"][cm.metadata.labels[LABEL]] = json.loads(state)
    try:
        raw["pool"] = co.get_cluster_custom_object(POOL_GROUP, POOL_VERSION, POOL_PLURAL, pool)
    except client.ApiException:
        raw["pool"] = None
    return raw


def _settings(s: dict[str, Any] | None) -> dict[str, Any] | None:
    if not s:
        return None
    pr = s.get("profiles") or {}
    return {"source": s.get("source"), "effectiveBatch": s.get("effectiveBatch"), "batch": s.get("batch"),
            "profileId": pr.get("profileId"), "throughput": pr.get("throughput"), "status": pr.get("status"),
            "reason": pr.get("reason") or s.get("rule"), "origin": pr.get("origin"),
            "evaluated": pr.get("evaluated"), "matched": pr.get("matched"), "feasible": pr.get("feasible")}


def _action(ra: dict[str, Any]) -> dict[str, Any]:
    spec, st = ra.get("spec") or {}, ra.get("status") or {}
    return {"name": ra["metadata"]["name"], "created": ra["metadata"].get("creationTimestamp"),
            "action": spec.get("action"), "target": spec.get("target"),
            **{k: st.get(k) for k in ACTION_KEYS}}


def _devices(pool: dict[str, Any] | None, owners: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for r in ((pool or {}).get("status") or {}).get("resources") or []:
        uuid = r.get("uuid") or ""
        out.append({"node": r.get("node"), "resource": r.get("resource"), "uuid": uuid, "model": r.get("model"),
                    "migProfile": r.get("migProfile") if r.get("deviceType") == "mig" else "",
                    "parent": r.get("parentUuid") or uuid, "holder": _holder(r), "workload": owners.get(uuid)})
    return sorted(out, key=lambda d: (d["node"] or "", d["parent"] or "", d["resource"] or ""))


def summarize(raw: dict[str, Any], samples: dict[str, list] | None = None) -> dict[str, Any]:
    by_plan: dict[str, list] = {}
    for ra in raw.get("runtimeactions") or []:
        by_plan.setdefault((ra.get("spec") or {}).get("executionPlanRef"), []).append(_action(ra))
    progress = raw.get("progress") or {}
    workloads, owners, linked = [], {}, set()
    for ep in raw.get("executionplans") or []:
        spec, st, name = ep.get("spec") or {}, ep.get("status") or {}, ep["metadata"]["name"]
        planning = spec.get("planning") or {}
        ranks = sorted(planning.get("rankMap") or [], key=lambda x: x["rank"])
        for x in ranks:
            if x.get("uuid"):
                owners[x["uuid"]] = name
        ref = planning.get("requestRef") or {}
        linked.add(ref.get("name"))
        state = progress.get(spec.get("workloadRef")) or {}
        workloads.append({
            "name": name, "workload": spec.get("workloadRef"), "runtime": spec.get("runtime"),
            "created": ep["metadata"].get("creationTimestamp"), "request": ref.get("name"),
            "state": st.get("state"), "validation": st.get("validation"), "message": st.get("message"),
            "generation": st.get("generation"), "parallelism": spec.get("parallelism"),
            "nodes": st.get("assignedNodes"), "binding": (st.get("binding") or {}).get("state"),
            "ranks": [{k: x.get(k) for k in ("rank", "dp", "tp", "pp", "node", "gpu", "uuid")} for x in ranks],
            "settings": _settings(planning.get("settings")), "rankSource": planning.get("rankSource"),
            "history": st.get("history") or [], "outcome": st.get("outcome"), "finishedAt": st.get("finishedAt"),
            "progress": {k: state.get(k) for k in PROGRESS_KEYS if k in state},
            "actions": sorted(by_plan.get(name, []), key=lambda a: a["created"] or ""),
            "samples": (samples or {}).get(name, [])})
    requests = [{"name": pr["metadata"]["name"], "created": pr["metadata"].get("creationTimestamp"),
                 "phase": (pr.get("status") or {}).get("phase"), "reason": (pr.get("status") or {}).get("reason"),
                 "parallelism": (pr.get("spec") or {}).get("parallelism")}
                for pr in raw.get("planrequests") or [] if pr["metadata"]["name"] not in linked]
    pool = raw.get("pool")
    return {"workloads": sorted(workloads, key=lambda w: w["created"] or "", reverse=True), "requests": requests,
            "pool": None if pool is None else {
                "name": pool["metadata"]["name"], "observedAt": (pool.get("status") or {}).get("observedAt"),
                "devices": _devices(pool, owners)}}
