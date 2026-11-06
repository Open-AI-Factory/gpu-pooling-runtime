from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import yaml

from . import placement
from ..runtime_coordination.adapter_manager.registry import ADAPTER_NAMES, capability_problems
from ..cli import ROOT
from .execution_plan import communication, device_binding, execution_settings, pool_topology, process_groups, uses_gpu
from ..models import Profiling, now
from .planner import MEMORY_HEADROOM, SELECTION_RULE, build_plan, enumerate_configs, evaluate
from ..pool import Candidate, MockPoolClient, is_pool_ref, load_pool

GROUP, VERSION = "grp.keti.re.kr", "v1alpha1"
ADAPTER_MODE = "k8s"
PLAN_RUNTIMES = ("ddp", "fsdp", "pytorch-3d", "deepspeed", "vllm")
DEFAULT_POOL = "mock/kind-snapshot.yaml"
LABEL_REQUEST, LABEL_WORKLOAD = "grp.keti.re.kr/request-id", "grp.keti.re.kr/workload"
_DNS = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")
_LABEL_VALUE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?$")
_PROFILING_FLOATS = ("weightsGb", "gradientsGb", "optimizerGb", "activationGbPerSample",
                     "modelMemoryGb", "activationMemoryGb")
_PROFILING_INTS = ("globalBatch", "numLayers", "numAttentionHeads")


@dataclass
class PlanRequest:
    name: str
    namespace: str
    request_id: str
    source: str
    workload: str
    runtime: str
    gpus: int
    gpus_per_node: int | None
    parallelism: dict[str, int] | None
    interconnect: str | None
    profiling: dict[str, Any]
    pool_snapshot: str
    expected_pool_version: str | None
    launch: dict[str, Any] | None
    allocation: list[str] | None
    digest: str
    reservation: dict[str, Any] | None = None
    awaiting: bool = False

    @property
    def key(self) -> str:
        return f"{self.namespace}/{self.workload}"


def _err(errors: list, fld: str, code: str, msg: str) -> None:
    errors.append({"field": fld, "code": code, "message": msg})


def _int(errors: list, fld: str, v: Any, minimum: int = 1) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int):
        _err(errors, fld, "InvalidType", f"{fld} must be an integer (got {v!r})")
        return None
    if v < minimum:
        _err(errors, fld, "OutOfRange", f"{fld} must be >= {minimum} (got {v})")
        return None
    return v


def parse_request(doc: Any, root: Path = ROOT) -> tuple[PlanRequest | None, list[dict]]:
    errors: list[dict] = []
    if not isinstance(doc, dict):
        return None, [{"field": "", "code": "InvalidType", "message": "request must be a mapping"}]
    if doc.get("kind", "PlanRequest") != "PlanRequest":
        _err(errors, "kind", "InvalidValue", f"kind must be PlanRequest (got {doc.get('kind')!r})")
    md, s = doc.get("metadata") or {}, doc.get("spec")
    if not isinstance(s, dict):
        return None, errors + [{"field": "spec", "code": "Required", "message": "spec is required"}]

    rid = s.get("requestId")
    if not rid:
        _err(errors, "spec.requestId", "Required", "requestId is required (used for duplicate detection)")
    elif not isinstance(rid, str) or not _LABEL_VALUE.match(rid):
        _err(errors, "spec.requestId", "InvalidValue",
             "requestId must be <=63 chars of [A-Za-z0-9_.-], starting and ending alphanumeric")
    workload = s.get("workloadRef")
    if not workload:
        _err(errors, "spec.workloadRef", "Required", "workloadRef is required")
    elif not isinstance(workload, str) or not _DNS.match(workload):
        _err(errors, "spec.workloadRef", "InvalidValue", "workloadRef must be a DNS-1123 label")
    runtime = s.get("runtime")
    if not runtime:
        _err(errors, "spec.runtime", "Required", "runtime is required")
    elif runtime not in PLAN_RUNTIMES:
        _err(errors, "spec.runtime", "InvalidValue", f"runtime must be one of {list(PLAN_RUNTIMES)} (got {runtime!r})")

    res = s.get("resources")
    gpus = gpn = None
    if not isinstance(res, dict) or "gpus" not in res:
        if not isinstance(s.get("parallelism"), dict):
            _err(errors, "spec.resources.gpus", "Required", "resources.gpus or parallelism is required")
    else:
        gpus = _int(errors, "spec.resources.gpus", res["gpus"])
        gpn = _int(errors, "spec.resources.gpusPerNode", res["gpusPerNode"]) if "gpusPerNode" in res else None
        if gpus and gpn and gpus % gpn:
            _err(errors, "spec.resources", "Inconsistent", f"gpus={gpus} is not a multiple of gpusPerNode={gpn}")

    par = s.get("parallelism")
    if par is not None:
        if not isinstance(par, dict):
            _err(errors, "spec.parallelism", "InvalidType", "parallelism must be a mapping {dp, tp, pp}")
            par = None
        else:
            vals = {k: _int(errors, f"spec.parallelism.{k}", par.get(k, 1)) for k in ("dp", "tp", "pp")}
            if all(vals.values()):
                par = vals
                if gpus is None and not any(e["field"].startswith("spec.resources") for e in errors):
                    gpus = vals["dp"] * vals["tp"] * vals["pp"]
                elif gpus and vals["dp"] * vals["tp"] * vals["pp"] != gpus:
                    _err(errors, "spec.parallelism", "Inconsistent",
                         f"DP×TP×PP={vals['dp'] * vals['tp'] * vals['pp']} ≠ resources.gpus={gpus}")
            else:
                par = None

    alloc = s.get("allocation")
    alloc_ids, reservation, awaiting = None, None, False
    if isinstance(alloc, dict) and "reservation" in alloc:
        reservation = _reservation(errors, alloc, gpus)
    elif isinstance(alloc, dict) and alloc.get("source") == "scheduler" and "gpuIds" not in alloc:
        awaiting = True
        if par is None:
            _err(errors, "spec.allocation.source", "Inconsistent",
                 "allocation from the scheduler needs spec.parallelism to describe the rank shape")
    elif alloc is not None:
        ids = alloc.get("gpuIds") if isinstance(alloc, dict) else None
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
            _err(errors, "spec.allocation.gpuIds", "InvalidType", "allocation.gpuIds must be a non-empty list of GPU ids")
        elif len(set(ids)) != len(ids):
            _err(errors, "spec.allocation.gpuIds", "InvalidValue", "allocation.gpuIds has duplicates")
        elif gpus and len(ids) != gpus:
            _err(errors, "spec.allocation.gpuIds", "Inconsistent",
                 f"allocation has {len(ids)} GPU(s) ≠ resources.gpus={gpus}")
        else:
            alloc_ids = ids

    prof = s.get("profiling") or {}
    if not isinstance(prof, dict):
        _err(errors, "spec.profiling", "InvalidType", "profiling must be a mapping")
        prof = {}
    for k in _PROFILING_FLOATS:
        if k in prof and (isinstance(prof[k], bool) or not isinstance(prof[k], (int, float)) or prof[k] < 0):
            _err(errors, f"spec.profiling.{k}", "InvalidValue", f"{k} must be a number >= 0 (got {prof[k]!r})")
    for k in _PROFILING_INTS:
        if k in prof and prof[k] is not None:
            _int(errors, f"spec.profiling.{k}", prof[k])

    pool_path = s.get("poolSnapshot", DEFAULT_POOL)
    if not isinstance(pool_path, str) or not (is_pool_ref(pool_path) or _resolve(pool_path, root).is_file()):
        _err(errors, "spec.poolSnapshot", "NotFound", f"pool snapshot not found: {pool_path!r}")
    launch = s.get("launch")
    if launch is not None and (not isinstance(launch, dict) or not launch.get("script")):
        _err(errors, "spec.launch", "InvalidValue", "launch must be a mapping with script")
    elif uses_gpu(launch) and not s.get("parallelism"):
        _err(errors, "spec.launch.gpu", "Inconsistent", "GPU launch requires spec.parallelism (one DRA claim per rank)")
    ic = s.get("interconnect")
    if ic is not None and not isinstance(ic, str):
        _err(errors, "spec.interconnect", "InvalidType", "interconnect must be a string")
    epv = s.get("expectedPoolVersion")
    if epv is not None and not isinstance(epv, str):
        _err(errors, "spec.expectedPoolVersion", "InvalidType", "expectedPoolVersion must be a string")

    name = md.get("name") or (f"{workload}-plan" if isinstance(workload, str) else "")
    if name and not _DNS.match(name):
        _err(errors, "metadata.name", "InvalidValue", "metadata.name must be a DNS-1123 label")
    if errors:
        return None, errors
    return PlanRequest(
        name=name, namespace=md.get("namespace", "default"), request_id=rid, source=s.get("source", "file"),
        workload=workload, runtime=runtime, gpus=gpus, gpus_per_node=gpn, parallelism=par,
        interconnect=ic, profiling=prof, pool_snapshot=pool_path, expected_pool_version=epv,
        launch=launch, allocation=alloc_ids, digest=request_digest(doc), reservation=reservation,
        awaiting=awaiting,
    ), []


_SLOT_KEYS = ("node_id", "gpu_id", "profile", "placement")


def _reservation(errors: list, alloc: dict, gpus: int | None) -> dict | None:
    r, fld = alloc["reservation"], "spec.allocation.reservation"
    if "gpuIds" in alloc:
        _err(errors, "spec.allocation", "Inconsistent", "give either allocation.gpuIds or allocation.reservation")
        return None
    if not isinstance(r, dict):
        _err(errors, fld, "InvalidType", "reservation must be a ReservationResult mapping")
        return None
    if r.get("ok") is not True:
        _err(errors, f"{fld}.ok", "InvalidValue", f"reservation was not granted: {r.get('reason') or 'ok is not true'}")
        return None
    slots = r.get("placements")
    if not isinstance(slots, list) or not slots or not all(
            isinstance(x, dict) and all(k in x for k in _SLOT_KEYS) and isinstance(x["placement"], int)
            for x in slots):
        _err(errors, f"{fld}.placements", "InvalidType",
             "placements must be a non-empty list of {node_id, gpu_id, profile, placement}")
        return None
    keys = [tuple(x[k] for k in _SLOT_KEYS) + (x.get("uuid"),) for x in slots]
    if len(set(keys)) != len(keys):
        _err(errors, f"{fld}.placements", "InvalidValue", "placements has duplicates")
        return None
    if gpus and len(keys) != gpus:
        _err(errors, f"{fld}.placements", "Inconsistent", f"reservation has {len(keys)} slot(s) ≠ resources.gpus={gpus}")
        return None
    return {"requestId": r.get("request_id"), "candidateId": r.get("candidate_id"), "slots": keys}


def _rid(doc: Any) -> str | None:
    s = doc.get("spec") if isinstance(doc, dict) else None
    rid = s.get("requestId") if isinstance(s, dict) else None
    return rid if isinstance(rid, str) else None


def request_digest(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc.get("spec"), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def load_request(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve(p: str, root: Path) -> Path | str:
    if is_pool_ref(p):
        return p
    return Path(p) if Path(p).is_absolute() else root / p


@dataclass
class PlanResult:
    request_id: str
    result: str
    reason: str = ""
    message: str = ""
    errors: list[dict] = field(default_factory=list)
    request_digest: str = ""
    decision: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    registration: dict[str, Any] | None = None
    time: str = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {"requestId": self.request_id, "result": self.result, "reason": self.reason,
                "message": self.message, "errors": self.errors, "requestDigest": self.request_digest,
                "time": self.time, "decision": self.decision, "registration": self.registration}

    @classmethod
    def from_dict(cls, d: dict, manifest: dict | None = None) -> "PlanResult":
        return cls(d.get("requestId", ""), d.get("result", ""), d.get("reason", ""), d.get("message", ""),
                   d.get("errors") or [], d.get("requestDigest", ""), d.get("decision"), manifest,
                   d.get("registration"), d.get("time", ""))


def _reject(rid: str, reason: str, msg: str, **kw) -> PlanResult:
    return PlanResult(rid, "Rejected", reason, msg, **kw)


def _pool_info(pool: MockPoolClient, path: str) -> dict[str, Any]:
    free = len(pool.free_gpus())
    return {"source": pool.source, "path": path, "version": pool.version, "atomicSnapshot": pool.atomic,
            "validUntil": pool.valid_until, "freeGpus": free, "excluded": pool.excluded}


def _divisors_desc(n: int) -> list[int]:
    return [d for d in range(n, 0, -1) if n % d == 0]


def _worker_shape(pool: MockPoolClient, req: PlanRequest) -> tuple[int | None, str]:
    if req.gpus_per_node:
        return req.gpus_per_node, "request"
    for d in _divisors_desc(req.gpus):
        if len(pool_topology(pool, d, req.key)["eligibleNodes"]) >= req.gpus // d:
            return d, "topology"
    return None, "topology"


def _given_allocation(pool: MockPoolClient, req: PlanRequest):
    by_id = {g.id: g for g in pool.gpus.values()}
    by_uuid = {g.physical: g for g in pool.gpus.values() if g.physical != g.id}
    gs, missing, taken = [], [], []
    for i in req.allocation:
        g = by_id.get(i) or by_uuid.get(i)
        if g is None:
            missing.append(i)
        elif g.allocated_to not in (None, req.key):
            taken.append(f"{i} (allocated to {g.allocated_to})")
        else:
            gs.append(g)
    if missing:
        excl = f"; excluded in pool: {[x['device'] for x in pool.excluded]}" if pool.excluded else ""
        return None, "InvalidSpec", f"allocated GPU(s) not in pool {pool.version}: {missing}{excl}", [
            {"field": "spec.allocation.gpuIds", "code": "NotFound", "message": f"not in pool: {missing}"}]
    if taken:
        return None, "InsufficientResources", f"allocated GPU(s) already in use per pool {pool.version}: {taken}", []
    return gs, "", "", []


def _placement_needs(gpus: int, prof: Profiling, cand, shard: bool,
                     adapter_rule: Callable[[int, int, int], str | None]) -> list[dict[str, Any]]:
    out = []
    for dp, tp, pp in enumerate_configs(gpus):
        if adapter_rule(dp, tp, pp):
            continue
        if (prof.num_attention_heads and prof.num_attention_heads % tp) or \
                (prof.num_layers and pp > 1 and prof.num_layers % pp):
            continue
        _, _, mem = evaluate(dp, tp, pp, prof, cand, shard)
        out.append({"dp": dp, "tp": tp, "pp": pp, "minGpusPerNode": tp,
                    "minGpuMemoryGb": round(mem * cand.ranks_per_physical / MEMORY_HEADROOM, 2)})
    return sorted(out, key=lambda n: (n["minGpuMemoryGb"], n["minGpusPerNode"]))[:3]


def _plan_given(req: PlanRequest, pool: MockPoolClient, decision: dict[str, Any]) -> PlanResult:
    p, dg = req.parallelism, req.digest
    prof, shard = Profiling.from_dict(req.profiling), req.runtime == "fsdp"
    decision.update(mode="Given", selectionRule=placement.RULE,
                    gpuCount={"value": req.gpus, "source": "parallelism", "analyzed": False})
    only = None
    if req.allocation:
        gs, reason, msg, errs = _given_allocation(pool, req)
        if reason:
            return _reject(req.request_id, reason, msg, errors=errs, request_digest=dg, decision=decision)
        only = {g.id for g in gs}
        decision["constraint"] = {"pinned": True, "gpuIds": sorted(only)}
    cands, free = placement.candidates(pool, req.key, p["dp"], p["tp"], p["pp"], only)
    decision["topology"] = {"usableGpusPerNode": free, "excludedDevices": len(pool.excluded),
                            "interconnect": {n: pool.interconnect.get(n, "unknown") for n in free}}
    if not req.profiling:
        decision["warnings"].append("profiling not provided: memory check uses 0 GB")
    if not cands:
        return _reject(req.request_id, "InsufficientResources",
                       f"pool {pool.version} cannot host {req.gpus} GPU(s) for DP{p['dp']} TP{p['tp']} PP{p['pp']}: "
                       f"usable GPUs per node {free or '{}'}"
                       + (f", {len(pool.excluded)} device(s) excluded" if pool.excluded else ""),
                       request_digest=dg, decision=decision)
    best = cands[0]
    cand = Candidate(best["gpuIds"], best["nodes"], req.interconnect or ("Network" if len(best["nodes"]) > 1 else
                     pool.interconnect.get(best["nodes"][0], "unknown")),
                     max(best["gpusPerNode"], p["tp"]), best["memoryGb"], ranks_per_physical=best["ranksPerPhysical"])
    ok, why, mem = evaluate(p["dp"], p["tp"], p["pp"], prof, cand, shard)
    row = {**p, "memPerGpuGb": round(mem, 2), "feasible": ok, "note": why}
    decision["candidates"] = [row]
    decision["placementCandidates"] = [placement.summary(c) for c in cands]
    if not ok:
        return _reject(req.request_id, "NoFeasiblePlan", f"DP{p['dp']} TP{p['tp']} PP{p['pp']}: {why}",
                       request_digest=dg, decision=decision)
    row["selected"] = True
    exec_problems = capability_problems(ADAPTER_MODE, req.runtime, p["tp"], p["pp"])
    decision["execution"] = {"executable": not exec_problems,
                             "reason": "; ".join(exec_problems) or f"{ADAPTER_MODE} adapter can run this plan"}
    nodes = best["nodes"]
    world = req.gpus
    decision["workers"] = {"gpus": req.gpus, "nodes": len(nodes), "gpusPerNode": best["gpusPerNode"],
                           "gpusPerNodeSource": "placement", "interconnect": cand.interconnect,
                           "memoryGbPerGpu": best["memoryGb"], "world": world,
                           "rankSlots": [{"rank": x["rank"], "nodeSlot": nodes.index(x["node"])} for x in best["rankMap"]],
                           "perNode": best["perNode"],
                           "constraints": [f"placement {best['label']}: TP groups {best['tpGroups']}, "
                                           f"PP stages {best['ppStages']}"]}
    decision["constraint"] = {"pinned": True, "gpuIds": best["gpuIds"], "nodes": nodes, "perNode": best["perNode"]}
    if best["tpCrossNode"]:
        decision["warnings"].append(f"allocated GPUs {best['perNode']} force {best['tpCrossNode']} TP group(s) "
                                    "across nodes")
    decision["rankMap"] = best["rankMap"]
    decision["processGroups"] = process_groups(best["rankMap"], p["dp"], p["tp"], p["pp"])
    decision["communication"] = communication(pool, nodes, req.launch)
    binding, unresolved = device_binding(best["rankMap"], req.launch)
    if unresolved:
        return _reject(req.request_id, "DeviceUnresolved",
                       f"DRA needs a device UUID per rank; pool {pool.version} has none for {unresolved}",
                       request_digest=dg, decision=decision)
    decision["deviceBinding"] = binding
    settings, why = execution_settings(pool, best["rankMap"], req.launch, p, req.namespace)
    if why:
        return _reject(req.request_id, "InvalidSpec", why, request_digest=dg, decision=decision)
    decision["settings"] = settings
    decision.update(selected=dict(p), reason=f"parallelism given by request; placement {best['label']} of "
                                              f"{len(cands)} candidate(s) selected by rule")
    return PlanResult(req.request_id, "Generated", "", f"ExecutionPlan {req.name} generated",
                      request_digest=dg, decision=decision, manifest=build_manifest(req, decision))


CANDIDATE_API = "keti.poolmanager/v0.1"


def allocation_request(req: PlanRequest) -> dict[str, Any]:
    p = req.parallelism
    world = p["dp"] * p["tp"] * p["pp"]
    _, _, mem = evaluate(p["dp"], p["tp"], p["pp"], Profiling.from_dict(req.profiling),
                         Candidate([], ["any"], "unknown", p["tp"], 1e9), req.runtime == "fsdp")
    return {"api_version": CANDIDATE_API, "kind": "CandidateRequest", "request_id": req.request_id,
            "workload": {"gpu_per_pod": 1, "replica": world, "allow_multi_node": world > 1,
                         "memory_per_gpu_gib": round(mem / MEMORY_HEADROOM, 2) if mem > 0 else None,
                         "whole_device": False, "vendor": None, "model": None, "partition_type": None},
            "max_candidates": None, "rankShape": dict(p)}


def generate(doc: dict, root: Path = ROOT, allow_stale_pool: bool = False, at=None,
             request_ref: dict[str, Any] | None = None) -> PlanResult:
    req, errors = parse_request(doc, root)
    if req is None:
        return _reject(_rid(doc) or "", "InvalidSpec",
                       f"{len(errors)} field error(s): " + ", ".join(e["field"] or "(root)" for e in errors),
                       errors=errors)
    dg = req.digest

    problems = capability_problems(ADAPTER_MODE, req.runtime)
    if problems and not req.parallelism:
        return _reject(req.request_id, "AdapterRejected", "; ".join(problems), request_digest=dg)

    try:
        pool = load_pool(_resolve(req.pool_snapshot, root))
    except Exception as e:
        return _reject(req.request_id, "PoolUnavailable", f"cannot read pool {req.pool_snapshot}: {e}",
                       request_digest=dg)
    info = _pool_info(pool, req.pool_snapshot)
    decision: dict[str, Any] = {
        "adapterMode": ADAPTER_MODE, "pool": info, "warnings": [],
        "gpuCount": {"value": req.gpus, "source": "request", "analyzed": False},
        "selectionRule": SELECTION_RULE,
    }
    if request_ref:
        decision["requestRef"] = request_ref
    if pool.stale(at):
        if not allow_stale_pool:
            return _reject(req.request_id, "PoolSnapshotStale",
                           f"pool status expired at {pool.valid_until}; re-read the pool", request_digest=dg,
                           decision=decision)
        decision["warnings"].append(f"pool status expired at {pool.valid_until} (allowed by --allow-stale-pool)")
    if req.awaiting:
        ask = allocation_request(req)
        decision["allocation"] = {"source": "scheduler", "state": "Awaiting"}
        decision["allocationRequest"] = ask
        return PlanResult(req.request_id, "AwaitingAllocation", "", f"waiting for the scheduler to reserve "
                          f"{ask['workload']['replica']} GPU slot(s); write spec.allocation.reservation",
                          request_digest=dg, decision=decision)
    if req.reservation:
        rv = req.reservation
        table = pool.slot_uuids()
        missing = ["{}/{}:{}@{}".format(*k) for k in rv["slots"] if not k[4] and k[:4] not in table]
        decision["allocation"] = {"source": "reservation", "state": "Reserved", "requestId": rv["requestId"],
                                  "candidateId": rv["candidateId"],
                                  "slots": ["{}/{}:{}@{}".format(*k) + (f" uuid={k[4]}" if k[4] else "")
                                            for k in rv["slots"]]}
        if missing:
            return _reject(req.request_id, "ReservationUnresolved",
                           f"reserved slot(s) have no DRA device in pool {pool.version}: {missing}",
                           errors=[{"field": "spec.allocation.reservation.placements", "code": "NotFound",
                                    "message": f"not in pool: {missing}"}],
                           request_digest=dg, decision=decision)
        req = replace(req, allocation=[k[4] or table[k[:4]] for k in rv["slots"]])
        decision["allocation"]["gpuIds"] = list(req.allocation)
    elif req.allocation:
        decision["allocation"] = {"source": "request", "gpuIds": list(req.allocation)}
    else:
        decision["allocation"] = {"source": "planner-fallback",
                                  "note": "no allocation from Pool Controller/Scheduler; GPUs chosen by this "
                                          "planner as a stand-in"}
        decision["warnings"].append("no allocation given: GPUs chosen by planner fallback, not by the scheduler")
    if req.expected_pool_version and req.expected_pool_version != pool.version:
        return _reject(req.request_id, "PoolSnapshotMismatch",
                       f"expected pool version {req.expected_pool_version}, current {pool.version}",
                       request_digest=dg, decision=decision)
    if req.parallelism:
        return _plan_given(req, pool, decision)

    pinned = None
    if req.allocation:
        gs, reason, msg, errs = _given_allocation(pool, req)
        if reason:
            return _reject(req.request_id, reason, msg, errors=errs, request_digest=dg, decision=decision)
        per_node: dict[str, int] = {}
        for g in gs:
            per_node[g.node] = per_node.get(g.node, 0) + 1
        pinned = {"gpuIds": [g.id for g in gs], "nodes": sorted(per_node), "perNode": per_node}
        decision["constraint"] = {"pinned": True, **pinned}
        cand = pool._cand(gs, "GPUs pinned by request")
        gpn, gpn_source, n_nodes = cand.gpus_per_node, "pinned", len(per_node)
        topo = {"usableGpusPerNode": per_node, "eligibleNodes": pinned["nodes"], "memoryGb": cand.memory_gb,
                "interconnect": cand.interconnect, "ranksPerPhysical": cand.ranks_per_physical}
    else:
        gpn, gpn_source = _worker_shape(pool, req)
        topo = pool_topology(pool, gpn or 1, req.key)
        n_nodes = req.gpus // gpn if gpn else 0
        if gpn is None or len(topo["eligibleNodes"]) < n_nodes:
            need = f"{n_nodes} node(s) x {gpn} GPU" if gpn else f"{req.gpus} GPU(s) in any node layout"
            msg = (f"pool {pool.version} cannot host {need}: usable GPUs per node {topo['usableGpusPerNode'] or '{}'}"
                   + (f", {len(pool.excluded)} device(s) excluded" if pool.excluded else ""))
            decision["topology"] = {**topo, "excludedDevices": len(pool.excluded)}
            return _reject(req.request_id, "InsufficientResources", msg, request_digest=dg, decision=decision)
        cand = Candidate([], [f"node{i}" for i in range(n_nodes)],
                         topo["interconnect"] if n_nodes == 1 else "Network", gpn, topo["memoryGb"],
                         ranks_per_physical=topo["ranksPerPhysical"])
    if req.interconnect:
        cand.interconnect = req.interconnect
    decision["topology"] = {**topo, "excludedDevices": len(pool.excluded)}
    decision["workers"] = {"gpus": req.gpus, "nodes": n_nodes, "gpusPerNode": gpn, "gpusPerNodeSource": gpn_source,
                           "interconnect": cand.interconnect, "memoryGbPerGpu": cand.memory_gb}
    if not req.profiling:
        decision["warnings"].append("profiling not provided: memory check uses 0 GB")

    prof, shard = Profiling.from_dict(req.profiling), req.runtime == "fsdp"

    def adapter_rule(dp: int, tp: int, pp: int) -> str | None:
        return "; ".join(capability_problems(ADAPTER_MODE, req.runtime, tp, pp)) or None

    def infeasible(msg: str) -> PlanResult:
        decision["feedback"] = {"options": _placement_needs(req.gpus, prof, cand, shard, adapter_rule)}
        return _reject(req.request_id, "NoFeasiblePlan", msg, request_digest=dg, decision=decision)

    if pinned and len(set(pinned["perNode"].values())) > 1:
        decision.update(mode="Selected", candidates=[])
        return infeasible(f"uneven pinned GPUs {pinned['perNode']}: {ADAPTER_MODE} Adapter needs the same GPU count per node")

    plan = build_plan(req.gpus, prof, cand, ADAPTER_NAMES.get(req.runtime, req.runtime), shard, adapter_rule)
    if plan is None:
        rows = []
        for dp, tp, pp in enumerate_configs(req.gpus):
            ok, why, mem = evaluate(dp, tp, pp, prof, cand, shard)
            rule = adapter_rule(dp, tp, pp)
            rows.append({"dp": dp, "tp": tp, "pp": pp, "memPerGpuGb": round(mem, 2),
                         "feasible": False, "note": rule if ok and rule else why})
        decision.update(mode="Selected", candidates=rows)
        return infeasible(f"none of {len(rows)} DP/TP/PP combination(s) for {req.gpus} GPU(s) "
                          f"({n_nodes} node(s) x {gpn}) passed the rules")
    decision.update(mode="Selected", candidates=plan.candidates)
    sel, reason = {"dp": plan.dp, "tp": plan.tp, "pp": plan.pp}, plan.reason
    decision.update(selected=sel, reason=reason)
    world = sel["dp"] * sel["tp"] * sel["pp"]
    constraints = [f"{gpn} GPU(s) of each node slot on the same node"]
    if sel["tp"] > 1:
        constraints.append(f"TP group of {sel['tp']} ranks within one node slot")
    if pinned:
        constraints.append(f"pinned to node(s) {pinned['nodes']}")
    decision["workers"].update(world=world, rankSlots=[{"rank": r, "nodeSlot": r // gpn} for r in range(world)],
                               constraints=constraints)
    return PlanResult(req.request_id, "Generated", "", f"ExecutionPlan {req.name} generated",
                      request_digest=dg, decision=decision, manifest=build_manifest(req, decision))


def build_manifest(req: PlanRequest, decision: dict[str, Any]) -> dict[str, Any]:
    sel, w = decision["selected"], decision["workers"]
    spec: dict[str, Any] = {
        "workloadRef": req.workload, "runtime": req.runtime,
        "parallelism": dict(sel),
        "placement": {"nodes": w["nodes"], "gpusPerNode": w["gpusPerNode"],
                      **({"perNode": w["perNode"]} if len(set((w.get("perNode") or {}).values())) > 1 else {})},
        "interconnect": w["interconnect"], "profiling": req.profiling, "poolSnapshot": req.pool_snapshot,
    }
    executable = (decision.get("execution") or {}).get("executable", True)
    if req.launch and executable:
        spec["launch"] = req.launch
    spec["planning"] = {
        "requestId": req.request_id, "source": req.source, "requestDigest": req.digest,
        "mode": decision["mode"], "poolVersion": decision["pool"]["version"],
        "gpuCount": decision["gpuCount"], "topology": decision["topology"], "workers": w,
        "selectionRule": decision["selectionRule"],
        "selected": dict(sel), "reason": decision["reason"], "candidates": decision["candidates"],
    }
    if decision.get("constraint"):
        spec["planning"]["pinnedNodes"] = decision["constraint"]["nodes"]
        spec["planning"]["pinnedGpuIds"] = decision["constraint"]["gpuIds"]
    for k in ("rankMap", "placementCandidates", "execution", "processGroups", "communication", "allocation",
              "deviceBinding", "settings"):
        if decision.get(k) is not None:
            spec["planning"][k] = decision[k]
    if not executable:
        spec["planning"]["planOnly"] = True
    md: dict[str, Any] = {"name": req.name, "namespace": req.namespace,
                          "labels": {LABEL_REQUEST: req.request_id, LABEL_WORKLOAD: req.workload},
                          "annotations": {"grp.keti.re.kr/plan-reason": decision["reason"]}}
    ref = decision.get("requestRef")
    if ref:
        spec["planning"]["requestRef"] = ref
        if ref.get("uid") and ref.get("namespace", req.namespace) == req.namespace:
            md["ownerReferences"] = [{"apiVersion": f"{GROUP}/{VERSION}", "kind": ref["kind"],
                                      "name": ref["name"], "uid": ref["uid"]}]
    return {"apiVersion": f"{GROUP}/{VERSION}", "kind": "ExecutionPlan", "metadata": md, "spec": spec}


def record(res: PlanResult, runs_dir: Path) -> Path | None:
    if not res.request_id:
        return None
    d = runs_dir / res.request_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.yaml").write_text(yaml.safe_dump(res.to_dict(), allow_unicode=True, sort_keys=False),
                                   encoding="utf-8")
    if res.manifest:
        (d / "executionplan.yaml").write_text(yaml.safe_dump(res.manifest, allow_unicode=True, sort_keys=False),
                                              encoding="utf-8")
    return d


def previous(doc: dict, runs_dir: Path) -> PlanResult | None:
    rid = _rid(doc)
    if not rid or not _LABEL_VALUE.match(rid):
        return None
    f = runs_dir / rid / "result.yaml"
    if not f.is_file():
        return None
    old = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    if old.get("result") not in ("Generated", "Registered"):
        return None
    if old.get("requestDigest") != request_digest(doc):
        return _reject(rid, "DuplicateRequest",
                       f"requestId {rid} was already used with different content (at {old.get('time')})",
                       request_digest=request_digest(doc))
    mf = runs_dir / rid / "executionplan.yaml"
    manifest = yaml.safe_load(mf.read_text(encoding="utf-8")) if mf.is_file() else None
    return PlanResult.from_dict(old, manifest)


Runner = Callable[..., subprocess.CompletedProcess]


class ClusterError(Exception):
    pass


class KubectlClient:
    def __init__(self, runner: Runner = subprocess.run):
        self.runner = runner

    def list_plans(self, namespace: str) -> list[dict]:
        out = self.runner(["kubectl", "get", "executionplans.grp.keti.re.kr", "-n", namespace, "-o", "json"],
                          capture_output=True, text=True)
        if out.returncode != 0:
            raise ClusterError(f"cannot list ExecutionPlans: {out.stderr.strip()}")
        return json.loads(out.stdout or "{}").get("items", [])

    def create_plan(self, manifest: dict) -> None:
        text = yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False)
        out = self.runner(["kubectl", "create", "-f", "-"], input=text, capture_output=True, text=True)
        if out.returncode != 0:
            raise ClusterError(out.stderr.strip() or "kubectl create failed")


class ApiClient:
    def __init__(self):
        from kubernetes import client
        self.co = client.CustomObjectsApi()

    def list_plans(self, namespace: str) -> list[dict]:
        try:
            return self.co.list_namespaced_custom_object(GROUP, VERSION, namespace, "executionplans")["items"]
        except Exception as e:
            raise ClusterError(f"cannot list ExecutionPlans: {e}") from e

    def create_plan(self, manifest: dict) -> None:
        try:
            self.co.create_namespaced_custom_object(GROUP, VERSION, manifest["metadata"]["namespace"],
                                                    "executionplans", manifest)
        except Exception as e:
            raise ClusterError(str(getattr(e, "body", None) or e)) from e


def register(res: PlanResult, root: Path = ROOT, allow_stale_pool: bool = False,
             runner: Runner = subprocess.run, at=None, client=None) -> PlanResult:
    client = client or KubectlClient(runner)
    mf = res.manifest
    md, spec = mf["metadata"], mf["spec"]
    ns, rid = md["namespace"], res.request_id
    base = dict(request_digest=res.request_digest, decision=res.decision, manifest=mf)

    try:
        items = client.list_plans(ns)
    except ClusterError as e:
        return _reject(rid, "RegistrationFailed", str(e), **base)
    for item in items:
        imd, ispec = item.get("metadata", {}), item.get("spec", {})
        other = (imd.get("labels") or {}).get(LABEL_REQUEST)
        if other == rid:
            return PlanResult(rid, "Duplicate", "DuplicateRequest",
                              f"already registered as ExecutionPlan {imd.get('name')}",
                              registration={"name": imd.get("name"), "namespace": ns, "existing": True}, **base)
        if imd.get("name") == md["name"] or ispec.get("workloadRef") == spec["workloadRef"]:
            return _reject(rid, "DuplicateRequest",
                           f"workload {spec['workloadRef']} already has ExecutionPlan {imd.get('name')} "
                           f"(request {other or '-'})", **base)

    try:
        pool = load_pool(_resolve(spec["poolSnapshot"], root))
    except Exception as e:
        return _reject(rid, "PoolUnavailable", f"cannot read pool {spec['poolSnapshot']}: {e}", **base)
    planned = res.decision["pool"]["version"]
    if pool.version != planned:
        return _reject(rid, "PoolSnapshotMismatch",
                       f"pool changed after decision ({planned} -> {pool.version}); generate again", **base)
    if pool.stale(at) and not allow_stale_pool:
        return _reject(rid, "PoolSnapshotStale", f"pool status expired at {pool.valid_until}", **base)

    try:
        client.create_plan(mf)
    except ClusterError as e:
        return _reject(rid, "RegistrationFailed", str(e), **base)
    return PlanResult(rid, "Registered", "", f"ExecutionPlan {md['name']} submitted to namespace {ns}",
                      registration={"time": now(), "name": md["name"], "namespace": ns}, **base)


def request_doc_from_resource(body: dict) -> dict:
    md = body.get("metadata") or {}
    return {"apiVersion": body.get("apiVersion"), "kind": body.get("kind", "PlanRequest"),
            "metadata": {"name": md.get("name"), "namespace": md.get("namespace", "default")},
            "spec": body.get("spec")}


def from_resource(body: dict, client, root: Path = ROOT) -> PlanResult:
    md = body.get("metadata") or {}
    doc = request_doc_from_resource(body)
    ref = {"kind": doc["kind"], "name": md.get("name"), "namespace": md.get("namespace", "default"),
           "uid": md.get("uid")}
    for _ in range(3):
        res = generate(doc, root, request_ref=ref)
        if res.result == "Generated":
            res = register(res, root, client=client)
        if res.reason != "PoolSnapshotMismatch":
            break
    return res


def status_summary(res: PlanResult) -> dict[str, Any]:
    d = res.decision or {}
    st: dict[str, Any] = {"result": res.result, "reason": res.reason, "message": res.message, "time": res.time,
                          "errors": res.errors}
    if res.result in ("Registered", "Duplicate"):
        st["executionPlan"] = (res.registration or {}).get("name") or (res.manifest or {}).get("metadata", {}).get("name")
    for k in ("gpuCount", "topology", "workers", "constraint", "mode", "selectionRule", "selected", "reason",
              "feedback", "execution", "allocation", "communication", "deviceBinding", "allocationRequest",
              "settings"):
        if d.get(k) is not None:
            st["decisionReason" if k == "reason" else k] = d[k]
    if d.get("candidates"):
        st["candidates"] = [f"DP{r['dp']} TP{r['tp']} PP{r['pp']} {'O' if r['feasible'] else 'X'} {r['note']}"
                            for r in d["candidates"]]
    if d.get("placementCandidates"):
        st["placementCandidates"] = [
            f"{c['label']}{' *' if c['selected'] else '  '} TP groups {c['tpGroups']}, PP {c['ppStages']}, "
            f"cross-node TP/DP/PP {c['tpCrossNode']}/{c['dpCrossNode']}/{c['ppCrossNode']}, "
            f"nodes {c['nodesUsed']}, fragmented {c['fragmentedNodes']}"
            for c in d["placementCandidates"]]
    if d.get("rankMap"):
        st["rankMap"] = [f"rank{x['rank']} DP{x['dp'] + 1} PP{x['pp'] + 1} TP{x['tp'] + 1} -> {x['gpu']}"
                         for x in d["rankMap"]]
    if d.get("processGroups"):
        st["processGroups"] = {k: " ".join(str(g) for g in v) for k, v in d["processGroups"].items()}
    if d.get("pool"):
        st["poolVersion"] = d["pool"]["version"]
    return st


def wait_validation(name: str, namespace: str, timeout: float, runner: Runner = subprocess.run,
                    interval: float = 1.0) -> dict[str, Any] | None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        out = runner(["kubectl", "get", "executionplans.grp.keti.re.kr", name, "-n", namespace, "-o", "json"],
                     capture_output=True, text=True)
        if out.returncode == 0:
            st = json.loads(out.stdout).get("status") or {}
            if st.get("validation"):
                return {k: st.get(k) for k in ("validation", "reason", "assignedNodes", "state")}
        time.sleep(interval)
    return None
