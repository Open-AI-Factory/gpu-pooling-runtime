from __future__ import annotations

import hashlib
import time
from pathlib import Path

from kubernetes import client, config

from .workload import LaunchError, Workload, to_host_path
from ..execution_plan_controller.execution_plan import rank_plan
from ..execution_plan_controller.profiles import brief
from ..runtime_coordination.adapter_manager.registry import REGISTRY
from ..models import ExecutionPlan, ParallelConfig, RuntimeActionSpec, WorkloadState, now
from .base import AdapterError, RuntimeAdapter

GROUP, VERSION, EP_PLURAL = "grp.keti.re.kr", "v1alpha1", "executionplans"
_loaded = False


def load_kube() -> None:
    global _loaded
    if not _loaded:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        _loaded = True


def _digest(path: Path) -> str:
    h = hashlib.sha256(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(2**20))
    return h.hexdigest()[:16]


def replacing(old, new):
    if isinstance(old, dict) and isinstance(new, dict):
        return {k: replacing(old.get(k), new[k]) if k in new else None for k in {**old, **new}}
    return new


class K8sAdapter(RuntimeAdapter):
    def __init__(self, spec: RuntimeActionSpec, pool):
        load_kube()
        self.name = spec.runtime
        self.pool = pool
        self.ns = spec.namespace
        self.ep_name = spec.execution_plan_ref or spec.adapter_options.get("executionPlan")
        if not self.ep_name:
            raise AdapterError("k8s mode requires executionPlanRef")
        self.co = client.CustomObjectsApi()
        self.wl = Workload(self.ns, spec.workload_ref.name)
        self._ready_samples: list[tuple[float, int]] = []

    def ep(self) -> dict:
        return self.co.get_namespaced_custom_object(GROUP, VERSION, self.ns, EP_PLURAL, self.ep_name)

    def patch_ep(self, spec: dict | None = None, status: dict | None = None) -> None:
        if spec:
            self.co.patch_namespaced_custom_object(GROUP, VERSION, self.ns, EP_PLURAL, self.ep_name,
                                                   {"spec": spec})
        if status:
            self.co.patch_namespaced_custom_object_status(GROUP, VERSION, self.ns, EP_PLURAL, self.ep_name,
                                                          {"status": status})

    def _requested(self) -> list[dict]:
        return (self.ep().get("status", {}).get("isolation", {}) or {}).get("requested", [])

    def current_state(self, spec):
        ep = self.ep()
        st, s = ep.get("status", {}), self.wl.state()
        if s.get("status") != "running":
            raise AdapterError(f"workload {self.wl.name} is not running (state={s.get('status')})")
        par = ep["spec"]["parallelism"]
        gpn = ep["spec"]["placement"].get("gpusPerNode", 1)
        ids = [g.id for g in self.pool.gpus_of(spec.workload_ref.key)]
        if not ids:
            ids = [f"{n}:{i}" for n in st.get("assignedNodes", []) for i in range(gpn)]
            self.pool.assign(spec.workload_ref.key, ids)
        return WorkloadState(ids, ParallelConfig(par["dp"], par["tp"], par["pp"]), {
            "runtime": spec.runtime, "mode": "k8s", "generation": st.get("generation"),
            "nodes": st.get("assignedNodes"), "worldSize": s.get("worldSize"), "step": s.get("step"),
            "loss": s.get("loss"), "rankPids": s.get("rankPids"), "rankPeakRssMb": s.get("rankPeakRssMb"),
            "perRankBatch": s.get("perRankBatch"), "globalBatch": s.get("globalBatch")})

    def prepare(self, spec, plan: ExecutionPlan):
        ep = self.ep()["spec"]
        ep_par = ep["parallelism"]
        if ep.get("runtime") in REGISTRY["k8s"].get("reshardRuntimes", []):
            return
        if (plan.dp, plan.tp, plan.pp) == (ep_par.get("dp", 1), ep_par.get("tp", 1), ep_par.get("pp", 1)):
            return
        if plan.tp > 1 or plan.pp > 1 or ep_par.get("tp", 1) > 1 or ep_par.get("pp", 1) > 1:
            raise AdapterError(f"{plan.parallel}: k8s Adapter cannot reconfigure TP/PP>1 "
                               "(checkpoint restore supports DP/FSDP only)")

    def preparation_ready(self, spec, plan):
        r = self.wl.readiness(self._requested())
        if r["state"] == "ReadyToRun":
            return True, "preparation=Ready"
        return False, r["detail"]

    def request_safe_point(self, spec):
        self.wl.channel.write("control", {"request": "safepoint", "action": spec.name})

    def safe_point_ready(self, spec):
        return self.wl.state().get("status") == "safepoint"

    def cancel_safe_point(self, spec):
        self.wl.channel.delete("control")
        time.sleep(1.0)
        s = self.wl.state()
        if s.get("status") == "safepoint":
            ep = self.ep()
            par = ep["spec"]["parallelism"]
            self._relaunch(spec, ParallelConfig(par["dp"], par["tp"], par["pp"]),
                           [g.id for g in self.pool.gpus_of(spec.workload_ref.key)],
                           s["checkpoint"], f"{spec.name}-keep", "safepoint cancelled, resume previous")

    def verify_checkpoint(self, spec):
        s = self.wl.state()
        try:
            p = to_host_path(s["checkpoint"])
            ok = p.exists() and p.stat().st_size == s.get("checkpointBytes") \
                and _digest(p) == s.get("checkpointDigest")
        except (KeyError, ValueError, OSError) as e:
            return False, f"serviceConfig=Invalid ({e})"
        if not ok:
            return False, "serviceConfig=Invalid (checkpoint size/digest mismatch)"
        launch = self.ep()["spec"].get("launch", {})
        if not to_host_path(f"/work/{launch.get('script', 'workloads/train_ddp.py')}").exists():
            return False, "serviceConfig=Invalid (training script not found)"
        return True, "serviceConfig=Valid"

    def _target_settings(self, spec) -> dict | None:
        cur = (self.ep()["spec"].get("launch") or {}).get("settings")
        if cur is None:
            return None
        return {**cur, **(spec.target_settings or {})}

    def plan_settings(self, spec, plan: ExecutionPlan):
        ep = self.ep()["spec"]
        previous = (ep.get("planning") or {}).get("settings")
        want = self._target_settings(spec)
        if not want:
            return None, previous, f"ExecutionPlan {self.ep_name} has no launch.settings to apply"
        planned, why = rank_plan(self.pool, spec.workload_ref.key, plan.dp, plan.tp, plan.pp, plan.gpu_ids,
                                 {**(ep.get("launch") or {}), "settings": want}, self.ns)
        if planned is None:
            return None, previous, why
        return planned["settings"], previous, ""

    def observe(self, spec):
        ep, s = self.ep(), self.wl.state()
        planned = (ep["spec"].get("planning") or {}).get("settings")
        applied = s.get("appliedSettings")
        match = None if not planned else applied == {"effectiveBatch": planned["effectiveBatch"], **planned["batch"]}
        return {"executionPlan": ep["metadata"]["name"], "generation": ep.get("status", {}).get("generation"),
                "planState": ep.get("status", {}).get("state"), "workloadStatus": s.get("status"),
                "step": s.get("step"), "worldSize": s.get("worldSize"), "loss": s.get("loss"),
                "nodes": ep.get("status", {}).get("assignedNodes"),
                "readiness": self.wl.readiness(self._requested())["state"],
                "plannedSettings": None if not planned else {
                    "source": planned.get("source"), "effectiveBatch": planned["effectiveBatch"],
                    "batch": planned["batch"], "profileId": (planned.get("profiles") or {}).get("profileId")},
                "appliedSettings": applied, "settingsMatch": match}

    def _relaunch(self, spec, par: ParallelConfig, gpu_ids: list[str], resume: str, tag: str,
                  history: str, settings: dict | None = None) -> int:
        ep = self.ep()
        world = par.gpus
        gen = int(ep.get("status", {}).get("generation", 0)) + 1
        launch = ep["spec"].get("launch") or {}
        launch_patch = None
        if settings is not None:
            launch_patch = {"settings": replacing(launch.get("settings") or {}, settings)}
            launch = {**launch, "settings": settings}
        plan, why = rank_plan(self.pool, spec.workload_ref.key, par.dp, par.tp, par.pp, gpu_ids, launch, self.ns)
        if plan is None:
            raise AdapterError(f"cannot rebuild execution plan: {why}")
        placed = [x["node"] for x in plan["rankMap"]]
        nodes, gpn = list(dict.fromkeys(placed)), plan["gpusPerNode"]
        self.wl.teardown()
        try:
            requested = self.wl.launch(world, placed, ep["spec"]["runtime"],
                                       launch, gen, resume, tag,
                                       parallelism={"dp": par.dp, "tp": par.tp, "pp": par.pp},
                                       owner={"apiVersion": f"{GROUP}/{VERSION}", "kind": "ExecutionPlan",
                                              "name": ep["metadata"]["name"], "uid": ep["metadata"]["uid"]},
                                       plan=plan)
        except LaunchError as e:
            raise AdapterError(str(e)) from e
        hist = list(ep.get("status", {}).get("history", []))
        hist.append({"time": now(), "action": spec.name, "event": history, "generation": gen,
                     "parallelism": {"dp": par.dp, "tp": par.tp, "pp": par.pp}, "nodes": nodes,
                     "gpuIds": list(gpu_ids), "settings": brief(plan["settings"])})
        planning = {k: plan[k] for k in ("selected", "rankMap", "processGroups", "communication", "deviceBinding",
                                         "settings")}
        planning["settings"] = replacing((ep["spec"].get("planning") or {}).get("settings"), plan["settings"])
        self.patch_ep(spec={"parallelism": {"dp": par.dp, "tp": par.tp, "pp": par.pp},
                            "placement": {"nodes": len(nodes), "gpusPerNode": gpn,
                                          "perNode": plan["perNode"] if len(set(plan["perNode"].values())) > 1 else None},
                            "planning": {**planning, "rankSource": "replan"},
                            **({"launch": launch_patch} if launch_patch else {})},
                      status={"generation": gen, "world": world, "assignedNodes": nodes,
                              "isolation": {"requested": requested}, "state": "Reconfiguring",
                              "history": hist[-10:]})
        return world

    def reconfigure(self, spec, plan: ExecutionPlan):
        sp = self.wl.state()
        self._sp = sp
        nodes = list(dict.fromkeys(self.pool.gpus[g].node for g in plan.gpu_ids))
        resume = sp["checkpoint"]
        if spec.adapter_options.get("failAt") == "Reconfigure":
            resume += ".missing"
        settings, event = None, f"reconfigure {sp.get('worldSize')}→{plan.parallel.gpus} ranks"
        if spec.action == "ApplySettings":
            self._settings_before = (self.ep()["spec"].get("launch") or {}).get("settings")
            settings, event = self._target_settings(spec), "apply execution settings"
        world = self._relaunch(spec, plan.parallel, plan.gpu_ids, resume, spec.name, event, settings)
        end = time.monotonic() + 180
        while time.monotonic() < end:
            s = self.wl.state()
            if s.get("status") == "running" and s.get("worldSize") == world and s.get("resumedStep") == sp["step"]:
                ws = WorkloadState(plan.gpu_ids, plan.parallel, {
                    "runtime": spec.runtime, "mode": "k8s", "nodes": nodes, "worldSize": world,
                    "step": s.get("step"), "loss": s.get("loss"), "rankPids": s.get("rankPids"),
                    "rankPeakRssMb": s.get("rankPeakRssMb"), "perRankBatch": s.get("perRankBatch"),
                    "globalBatch": s.get("globalBatch"), "resumedStep": s.get("resumedStep"),
                    "checkpointWorldSize": s.get("checkpointWorldSize"),
                    "checkpointStrategy": s.get("checkpointStrategy"), "checkpoint": sp["checkpoint"]})
                return ws
            failed = [p.metadata.name for p in self.wl.pods() if p.status.phase == "Failed"]
            if failed:
                raise AdapterError(f"restarted Pod failed: {failed}")
            time.sleep(1)
        raise AdapterError("timed out waiting for training to resume after restart (180s)")

    def verify_ready(self, spec):
        end = time.monotonic() + 60
        first = self.wl.state().get("step", 0)
        while time.monotonic() < end:
            r = self.wl.readiness(self._requested())
            s = self.wl.state()
            if r["state"] == "ReadyToRun" and s.get("status") == "running" and s.get("step", 0) > first:
                planned = (self.ep()["spec"].get("planning") or {}).get("settings")
                applied = s.get("appliedSettings")
                if planned and applied != {"effectiveBatch": planned["effectiveBatch"], **planned["batch"]}:
                    return False, f"ready=false settings mismatch: planned {planned['batch']} applied {applied}"
                self.patch_ep(status={"state": "ReadyToRun", "appliedSettings": applied})
                return True, "ready=true response=OK" + (f" settings=Applied {applied}" if planned else "")
            time.sleep(1)
        return False, f"ready=false ({self.wl.readiness(self._requested())['detail']})"

    def rollback(self, spec, before):
        sp = getattr(self, "_sp", None) or self.wl.state()
        if not sp.get("checkpoint"):
            raise AdapterError("no checkpoint to restore from")
        world = self._relaunch(spec, before.parallel, before.gpu_ids, sp["checkpoint"], f"{spec.name}-rollback",
                               "rollback to previous", getattr(self, "_settings_before", None))
        end = time.monotonic() + 180
        while time.monotonic() < end:
            s = self.wl.state()
            if s.get("status") == "running" and s.get("worldSize") == world:
                self.patch_ep(status={"state": "ReadyToRun"})
                return
            time.sleep(1)
        raise AdapterError("timed out waiting for resume after restore")
