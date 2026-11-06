from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import kopf
from kubernetes import client, config

from ...cli import ROOT
from .controller import RuntimeActionController
from ...execution_plan_controller.execution_plan import uses_gpu, validate as validate_plan
from ...execution_plan_controller.profiles import brief
from ...runtime_adapter.workload import GEN, LaunchError, Workload
from ...models import RuntimeActionSpec, now
from ...pool import is_pool_ref, load_pool

GROUP, VERSION = "grp.keti.re.kr", "v1alpha1"
RA, EP, PR = "runtimeactions", "executionplans", "planrequests"
log = logging.getLogger("grp_runtime")

_reg_lock = threading.Lock()
_wl_locks: dict[str, threading.Lock] = {}
_inflight: dict[str, list[dict]] = {}


def _abs(path: str | None) -> str | None:
    if not path or is_pool_ref(path) or Path(path).is_absolute():
        return path
    return str((ROOT / path).resolve())


def _claim_binding(spec: dict, wl: Workload, generation: int) -> dict:
    world = spec["parallelism"]["dp"] * spec["parallelism"]["tp"] * spec["parallelism"]["pp"]
    try:
        rows = wl.claim_bindings(generation)
    except Exception as e:
        return {"state": "Unknown", "reason": f"cannot read ResourceClaims: {e}", "checkedAt": now()}
    devices = [{"rank": x["rank"], "pod": (x["reservedFor"] or [None])[0], "node": x["node"],
                "resource": x["resource"], "uuid": x["uuid"], "claim": x["claim"]} for x in rows if x["allocated"]]
    bound = len(rows) == world and len(devices) == world
    waiting = [x["claim"] for x in rows if not x["allocated"]]
    return {"state": "Bound" if bound else "Pending", "pool": "dra", "devices": devices,
            "reason": "" if bound else f"claims not allocated: {waiting or f'{len(rows)}/{world} claims'}",
            "checkedAt": now()}


def _binding(spec: dict, namespace: str, wl: Workload, generation: int) -> dict | None:
    if not uses_gpu(spec.get("launch")):
        return None
    return _claim_binding(spec, wl, generation)


@kopf.on.startup()
def _startup(settings: kopf.OperatorSettings, **_):
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    settings.persistence.progress_storage = kopf.AnnotationsProgressStorage()
    settings.posting.level = logging.WARNING
    logging.getLogger("grp_runtime").setLevel(logging.INFO)


def _plan_request(body: dict, name: str) -> dict:
    from ...execution_plan_controller import intake
    res = intake.from_resource(body, intake.ApiClient())
    st = intake.status_summary(res)
    w = st.get("workers") or {}
    log.info("[%s] PlanRequest %s%s | GPUs=%s workers=%s x %s selected=%s %s", name, res.result,
             f" reason={res.reason}" if res.reason else "", w.get("gpus"), w.get("nodes"), w.get("gpusPerNode"),
             st.get("selected"), res.message)
    return st


@kopf.on.create(GROUP, VERSION, PR)
async def pr_create(body, name, namespace, patch, **_):
    if (body.get("status") or {}).get("result"):
        return
    await asyncio.to_thread(client.CustomObjectsApi().patch_namespaced_custom_object_status,
                            GROUP, VERSION, namespace, PR, name, {"status": {"phase": "Planned"}})
    st = await asyncio.to_thread(_plan_request, dict(body), name)
    if st.get("result") in ("Rejected", "AwaitingAllocation"):
        st["phase"] = st["result"]
    patch.status.update(st)


@kopf.on.update(GROUP, VERSION, PR, field="spec.allocation")
async def pr_allocated(body, name, patch, **_):
    if (body.get("status") or {}).get("result") != "AwaitingAllocation":
        return
    st = await asyncio.to_thread(_plan_request, dict(body), name)
    st["phase"] = {"Rejected": "Rejected", "AwaitingAllocation": "AwaitingAllocation"}.get(st.get("result"), "Planned")
    patch.status.update(st)


def _report(body: dict, namespace: str, status: dict) -> None:
    ref = (body["spec"].get("planning") or {}).get("requestRef") or {}
    if ref.get("kind") != "PlanRequest" or not ref.get("name"):
        return
    try:
        client.CustomObjectsApi().patch_namespaced_custom_object_status(
            GROUP, VERSION, ref.get("namespace", namespace), PR, ref["name"], {"status": status})
    except Exception as e:
        log.warning("[%s] cannot report to PlanRequest %s: %s", body["metadata"]["name"], ref["name"], e)


def _owner(body: dict) -> dict | None:
    md = body.get("metadata") or {}
    if not md.get("uid"):
        return None
    return {"apiVersion": f"{GROUP}/{VERSION}", "kind": "ExecutionPlan", "name": md["name"], "uid": md["uid"]}


def _validate_and_maybe_launch(body: dict, namespace: str, launch: bool) -> dict:
    spec, name = body["spec"], body["metadata"]["name"]
    pool = load_pool(_abs(spec.get("poolSnapshot", "mock/kind-snapshot.yaml")))
    st = body.get("status") or {}
    first = launch and not st.get("generation")
    res = validate_plan(body, pool, place_workers=first)
    status = {**res, "validatedAt": now()}
    if res["validation"] == "Rejected":
        log.info("[%s] ExecutionPlan Rejected: %s", name, res["reason"])
        return status
    pl = res.get("placement")
    if not first or pl is None:
        return status
    pl["time"] = now()
    retry = st.get("state") == "Unschedulable"
    if pl["state"] != "Placed":
        status["state"] = "Unschedulable"
        if not retry:
            _report(body, namespace, {"placement": pl})
            log.info("[%s] ExecutionPlan Accepted -> Unschedulable (retry every 2s): %s", name, pl["reason"])
        return status
    _report(body, namespace, {"placement": pl, **({"phase": "Running"} if spec.get("launch") else {})})
    if spec.get("launch"):
        par = spec["parallelism"]
        world = par["dp"] * par["tp"] * par["pp"]
        nodes = res["assignedNodes"]
        wl = Workload(namespace, spec["workloadRef"])
        try:
            requested = wl.launch(world, pl["rankNodes"], spec["runtime"], spec["launch"], 1, None, "initial",
                                  parallelism={k: par[k] for k in ("dp", "tp", "pp")}, owner=_owner(body),
                                  plan=spec.get("planning") if pl.get("rankSource") == "plan" else None)
        except LaunchError as e:
            status.update(generation=1, world=world, state="LaunchFailed", message=str(e))
            _report(body, namespace, {"phase": "LaunchFailed", "message": str(e)})
            log.warning("[%s] ExecutionPlan launch failed: %s", name, e)
            return status
        status.update(generation=1, world=world, state="Launching", isolation={"requested": requested},
                      history=[{"time": now(), "event": "launch", "generation": 1,
                                "parallelism": par, "nodes": nodes,
                                "settings": brief((spec.get("planning") or {}).get("settings"))}])
        log.info("[%s] ExecutionPlan Accepted -> placed by %s on %s -> launched %d rank Pod(s) (rank layout: %s)",
                 name, pl["by"], nodes, world, pl.get("rankSource"))
    elif (spec.get("planning") or {}).get("planOnly"):
        status["state"] = "PlanOnly"
        log.info("[%s] ExecutionPlan Accepted -> placed by %s on %s (plan only, not launched)",
                 name, pl["by"], res["assignedNodes"])
    return status


@kopf.on.create(GROUP, VERSION, EP)
async def ep_create(body, namespace, patch, **_):
    patch.status.update(await asyncio.to_thread(_validate_and_maybe_launch, dict(body), namespace, True))


@kopf.on.update(GROUP, VERSION, EP, field="spec")
async def ep_update(body, namespace, patch, **_):
    res = await asyncio.to_thread(_validate_and_maybe_launch, dict(body), namespace, False)
    patch.status.update(res)


def _ep_tick(body: dict, namespace: str) -> dict | None:
    st = body.get("status") or {}
    if not st.get("generation"):
        if st.get("validation") == "Accepted" and st.get("state") == "Unschedulable":
            upd = _validate_and_maybe_launch(body, namespace, True)
            if (upd.get("placement") or {}).get("state") == "Placed":
                return upd
        return None
    if st.get("state") in ("Succeeded", "Failed", "LaunchFailed"):
        return None
    workload = body["spec"]["workloadRef"]
    wl = Workload(namespace, workload)
    with _reg_lock:
        busy = any(x["running"] for x in _inflight.get(f"{namespace}/{workload}", []))
    done = None if busy else wl.outcome(st["generation"])
    if done:
        s = wl.state()
        outcome = {k: s.get(k) for k in ("step", "loss", "stepsPerSec", "globalBatch", "device", "mlflowRunId") if k in s}
        upd = {"state": done, "finishedAt": now(), "outcome": outcome}
        _report(body, namespace, {"phase": done, "finishedAt": upd["finishedAt"], "outcome": outcome})
        log.info("[%s] ExecutionPlan %s generation=%s %s", body["metadata"]["name"], done, st["generation"], outcome)
        return upd
    r = wl.readiness((st.get("isolation") or {}).get("requested", []))
    state = "Reconfiguring" if busy else r["state"]
    agents = {"hook": r["hook"], "ranks": r["ranks"], "world": r["world"], "isolation": r["isolation"],
              "mismatches": r["mismatches"], "staleRanks": r["staleRanks"]}
    upd: dict = {}
    old = st.get("binding") or {}
    if not busy and old.get("state") != "Bound":
        b = _binding(body["spec"], namespace, wl, st["generation"])
        if b and (b["state"], b.get("devices"), b.get("reason")) != (old.get("state"), old.get("devices"),
                                                                     old.get("reason")):
            upd["binding"] = b
            _report(body, namespace, {"binding": {k: b.get(k) for k in ("state", "pool", "reason", "checkedAt")}
                                      | {"devices": len(b.get("devices", []))}})
            if b["state"] == "Bound":
                log.info("[%s] resources Bound in pool %s: %s", body["metadata"]["name"], b["pool"],
                         [(d["rank"], d["resource"]) for d in b["devices"]])
    if st.get("state") != state or st.get("agents") != agents:
        upd.update(state=state, agents=agents)
    return upd or None


@kopf.timer(GROUP, VERSION, EP, interval=2.0, idle=0)
async def ep_timer(body, namespace, patch, **_):
    upd = await asyncio.to_thread(_ep_tick, dict(body), namespace)
    if upd:
        patch.status.update(upd)


def _fill_from_plan(spec: RuntimeActionSpec, namespace: str) -> None:
    co = client.CustomObjectsApi()
    ep = co.get_namespaced_custom_object(GROUP, VERSION, namespace, EP, spec.execution_plan_ref)
    es = ep["spec"]
    spec.runtime = spec.runtime or es["runtime"]
    spec.workload_ref.name = es["workloadRef"]
    spec.workload_ref.namespace = namespace
    spec.adapter_mode = "k8s"
    spec.adapter_options = {**spec.adapter_options, "executionPlan": spec.execution_plan_ref}
    spec.pool_snapshot = _abs(es.get("poolSnapshot", "mock/kind-snapshot.yaml"))
    if not any(v for v in vars(spec.profiling).values() if v not in (None, 0.0, "training")):
        from ...models import Profiling
        spec.profiling = Profiling.from_dict(es.get("profiling"))


def _run_action(spec: RuntimeActionSpec, name: str, namespace: str, logger) -> None:
    api = client.CustomObjectsApi()

    def patch(st):
        api.patch_namespaced_custom_object_status(GROUP, VERSION, namespace, RA, name, {"status": st.to_dict()})

    key = spec.workload_ref.key
    me = {"name": name, "action": spec.action, "target": spec.target_gpu_count, "running": False}
    with _reg_lock:
        queue = _inflight.setdefault(key, [])
        dup = next((x for x in queue if x["action"] == me["action"] and x["target"] == me["target"]), None)
        if dup is None:
            queue.append(me)
            ahead = [x["name"] for x in queue if x is not me]
        lock = _wl_locks.setdefault(key, threading.Lock())
    if dup is not None:
        msg = f"{dup['name']} with the same workload and target (GPUs {me['target']}) is running or queued"
        patch_body = {"phase": "Aborted", "mode": spec.mode, "result": "reason=DuplicateRequest previous=Kept",
                      "reason": "DuplicateRequest", "previous": "Kept", "message": msg,
                      "conditions": [{"phase": "Accepted", "time": now(), "result": "request validated",
                                      "reason": "", "message": msg},
                                     {"phase": "Aborted", "time": now(),
                                      "result": "reason=DuplicateRequest previous=Kept",
                                      "reason": "DuplicateRequest", "message": msg}]}
        api.patch_namespaced_custom_object_status(GROUP, VERSION, namespace, RA, name, {"status": patch_body})
        log.info("[%s] Aborted reason=DuplicateRequest previous=Kept | %s", name, msg)
        return
    history = []
    if ahead:
        history = [{"phase": "Accepted", "time": now(), "result": "queued", "reason": "",
                    "message": f"waiting for preceding action(s) {ahead}"}]
        api.patch_namespaced_custom_object_status(GROUP, VERSION, namespace, RA, name, {"status": {
            "phase": "Accepted", "mode": spec.mode, "result": "queued",
            "message": history[0]["message"], "conditions": history}})
        log.info("[%s] Accepted queued | preceding %s", name, ahead)
    try:
        with lock:
            with _reg_lock:
                me["running"] = True
            RuntimeActionController(ROOT / "runs", on_transition=patch, poll_interval=0.5).run(
                spec, history=history)
    finally:
        with _reg_lock:
            queue.remove(me)


@kopf.on.create(GROUP, VERSION, RA)
async def ra_create(body, name, namespace, logger, **_):
    if (body.get("status") or {}).get("phase"):
        return
    spec = RuntimeActionSpec.from_manifest(dict(body))
    spec.namespace = namespace
    if spec.execution_plan_ref:
        await asyncio.to_thread(_fill_from_plan, spec, namespace)
    spec.pool_snapshot = _abs(spec.pool_snapshot)
    await asyncio.to_thread(_run_action, spec, name, namespace, logger)
