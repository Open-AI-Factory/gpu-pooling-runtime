from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from ..adapter_manager.registry import AdapterError, select_adapter
from ...models import (ACTIONS, KEEP_LAYOUT_ACTIONS, RUNTIMES, ExecutionPlan, Phase, Previous, RuntimeActionSpec,
                       RuntimeActionStatus, now)
from ...execution_plan_controller.planner import build_plan, enumerate_configs, evaluate
from ...pool import CandidateSet, Gpu, MockPoolClient, load_pool

log = logging.getLogger("grp_runtime")


def validate(spec: RuntimeActionSpec) -> list[str]:
    errs = []
    if not spec.workload_ref.name:
        errs.append("workloadRef.name missing")
    if spec.runtime not in RUNTIMES:
        errs.append(f"runtime '{spec.runtime}' not supported {RUNTIMES}")
    if spec.action not in ACTIONS:
        errs.append(f"action '{spec.action}' not supported {ACTIONS}")
    if spec.target_gpu_count < 1 and spec.action not in KEEP_LAYOUT_ACTIONS:
        errs.append("target.gpuCount must be >= 1")
    if spec.action == "Observe" and (spec.target_parallelism or spec.target_gpu_ids or spec.target_settings):
        errs.append("Observe changes nothing: target.parallelism, allocation and settings are not allowed")
    if spec.target_settings is not None and (spec.action != "ApplySettings" or not isinstance(spec.target_settings, dict)):
        errs.append("target.settings must be an object and is only used by ApplySettings")
    if spec.safe_policy.mode not in ("WaitSafePoint", "Immediate"):
        errs.append(f"safePolicy.mode '{spec.safe_policy.mode}' not supported")
    if spec.mode != "restart":
        errs.append(f"mode '{spec.mode}' not supported (only restart)")
    if not spec.pool_snapshot:
        errs.append("poolSnapshot missing (mock snapshot required until resource pool integration)")
    par = spec.target_parallelism
    if par is not None and spec.action == "ApplySettings":
        pass
    elif par is not None:
        if not isinstance(par, dict) or any(not isinstance(par.get(k), int) or par[k] < 1 for k in ("dp", "tp", "pp")):
            errs.append("target.parallelism needs integer dp, tp, pp >= 1")
        elif par["dp"] * par["tp"] * par["pp"] != spec.target_gpu_count:
            errs.append(f"target.parallelism DP×TP×PP={par['dp'] * par['tp'] * par['pp']} ≠ target.gpus={spec.target_gpu_count}")
    ids = spec.target_gpu_ids
    if ids is not None:
        if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
            errs.append("target.allocation.gpuIds must be a list of distinct GPU ids")
        elif spec.action not in KEEP_LAYOUT_ACTIONS and len(ids) != spec.target_gpu_count:
            errs.append(f"target.allocation has {len(ids)} GPU(s) ≠ target.gpus={spec.target_gpu_count}")
    return errs


def given_gpus(pool: MockPoolClient, key: str, ids: list[str]) -> tuple[list[Gpu], str]:
    found = {}
    for g in pool.gpus.values():
        for k in (g.id, g.uuid, g.physical):
            if k:
                found.setdefault(k, g)
    missing = [i for i in ids if i not in found]
    if missing:
        return [], f"allocated GPU(s) not in pool {pool.version}: {missing}"
    gs = [found[i] for i in ids]
    taken = [f"{i} (allocated to {g.allocated_to})" for i, g in zip(ids, gs) if g.allocated_to not in (None, key)]
    if taken:
        return [], f"allocated GPU(s) already in use per pool {pool.version}: {taken}"
    return gs, ""


class RuntimeActionController:
    def __init__(self, runs_root: Path, on_transition: Callable[[RuntimeActionStatus], None] | None = None,
                 poll_interval: float = 0.1):
        self.runs_root = runs_root
        self.on_transition = on_transition
        self.poll = poll_interval

    def _enter(self, st: RuntimeActionStatus, spec, phase: Phase, result: str, msg: str = "",
               reason: str = "") -> None:
        prev = st.phase.value if st.conditions else "-"
        st.phase, st.result, st.message = phase, result, msg
        if reason:
            st.reason = reason
        st.conditions.append({"phase": phase.value, "time": now(), "result": result,
                              "reason": reason, "message": msg})
        log.info("[%s] %s → %-11s %s%s", spec.name, prev, phase.value, result,
                 f" | {msg}" if msg else "")
        if self.on_transition:
            self.on_transition(st)

    def _abort(self, st, spec, reason: str, msg: str, previous: Previous = Previous.KEPT):
        st.previous = previous.value
        self._enter(st, spec, Phase.ABORTED, f"reason={reason} previous={previous.value}", msg, reason)
        return st

    def run(self, spec: RuntimeActionSpec, pool: MockPoolClient | None = None,
            history: list[dict] | None = None) -> RuntimeActionStatus:
        st = RuntimeActionStatus(mode=spec.mode, conditions=list(history or []))

        errs = validate(spec)
        if errs:
            self._enter(st, spec, Phase.ACCEPTED, "request invalid",
                        f"{spec.action} {spec.workload_ref.key} → GPU {spec.target_gpu_count}")
            return self._abort(st, spec, "InvalidSpec", "; ".join(errs))
        goal = "current GPUs" if spec.action in KEEP_LAYOUT_ACTIONS else f"GPU {spec.target_gpu_count}"
        self._enter(st, spec, Phase.ACCEPTED, "request validated",
                    f"{spec.action} {spec.workload_ref.key} → {goal}, mode={spec.mode}")
        pool = pool or load_pool(spec.pool_snapshot)
        deadline = time.monotonic() + spec.safe_policy.timeout_seconds

        try:
            adapter = select_adapter(spec, pool, self.runs_root)
        except AdapterError as e:
            return self._abort(st, spec, "AdapterRejected", str(e))
        st.adapter = adapter.name
        self._enter(st, spec, Phase.PREPARING, "preparation=InProgress", f"adapter={adapter.name}")
        try:
            before = adapter.current_state(spec)
        except AdapterError as e:
            return self._abort(st, spec, "WorkloadNotFound", str(e))
        st.before = before.to_dict()
        cur = len(before.gpu_ids)
        if spec.action == "Observe":
            return self._observe(st, spec, adapter)
        if spec.action == "ApplySettings":
            return self._apply_settings(st, spec, adapter, pool, before, deadline)
        if spec.action == "ScaleDown" and spec.target_gpu_count >= cur:
            return self._abort(st, spec, "InvalidAction",
                               f"ScaleDown but target GPUs {spec.target_gpu_count} >= current {cur}")
        if spec.action == "ScaleUp" and spec.target_gpu_count <= cur:
            return self._abort(st, spec, "InvalidAction",
                               f"ScaleUp but target GPUs {spec.target_gpu_count} <= current {cur}")
        if spec.action == "Resize" and spec.target_gpu_count == cur:
            return self._abort(st, spec, "InvalidAction", f"target GPUs {cur} = current GPUs, nothing to change")

        if spec.target_gpu_ids:
            gs, why = given_gpus(pool, spec.workload_ref.key, spec.target_gpu_ids)
            if why:
                return self._abort(st, spec, "InsufficientResources", why)
            cand = pool._cand(gs, "GPUs given by target.allocation")
            cand.rank = 1
            cs = CandidateSet(pool.version, [cand])
            allocation = {"source": "request", "gpuIds": list(spec.target_gpu_ids)}
        else:
            cs = pool.get_candidates(spec.workload_ref.key, before.gpu_ids, spec.target_gpu_count)
            allocation = {"source": "planner-fallback",
                          "note": "no target.allocation from Pool Controller/Scheduler; GPUs chosen by this "
                                  "controller as a stand-in"}
        if not cs.candidates:
            return self._abort(st, spec, "InsufficientResources",
                               f"no candidate for {spec.target_gpu_count} GPU(s) in resource pool (snapshot {cs.snapshot_version})")
        given = spec.target_parallelism
        exclude = (lambda dp, tp, pp: None if (dp, tp, pp) == (given["dp"], given["tp"], given["pp"])
                   else "parallelism given by target") if given else None
        plan, rejected = None, []
        shard = spec.runtime == "fsdp"
        for cand in cs.candidates:
            plan = build_plan(spec.target_gpu_count, spec.profiling, cand, adapter.name, shard, exclude)
            if plan:
                plan.reason = f"[candidate#{cand.rank}: {cand.reason}] {plan.reason}"
                break
            notes = [f"DP{dp}/TP{tp}/PP{pp}: {evaluate(dp, tp, pp, spec.profiling, cand, shard)[1]}"
                     for dp, tp, pp in enumerate_configs(spec.target_gpu_count)]
            rejected.append(f"candidate#{cand.rank}({'; '.join(notes)})")
        if plan is None:
            return self._abort(st, spec, "NoFeasiblePlan",
                               f"no feasible DP/TP/PP combination: {' / '.join(rejected)}")
        st.execution_plan = {"snapshotVersion": cs.snapshot_version, **plan.to_dict(), "allocation": allocation,
                             "parallelismSource": "target" if given else "planner"}
        return self._execute(st, spec, adapter, pool, before, plan, deadline)

    def _observe(self, st, spec, adapter):
        try:
            obs = adapter.observe(spec)
        except AdapterError as e:
            return self._abort(st, spec, "WorkloadNotFound", str(e))
        st.observation = {"observedAt": now(), **obs}
        match = obs.get("settingsMatch")
        result = "observed" + ("" if match is None else f" settings={'Applied' if match else 'Mismatch'}")
        self._enter(st, spec, Phase.SUCCEEDED, result,
                    f"{st.before['dp']}/{st.before['tp']}/{st.before['pp']} (DP/TP/PP) on {st.before['gpuCount']} GPU(s), "
                    "no change made")
        return st

    def _apply_settings(self, st, spec, adapter, pool, before, deadline):
        cur = len(before.gpu_ids)
        par = spec.target_parallelism
        if spec.target_gpu_count and spec.target_gpu_count != cur:
            return self._abort(st, spec, "InvalidAction",
                               f"ApplySettings keeps the GPUs: target GPUs {spec.target_gpu_count} != current {cur}")
        if par and (par.get("dp"), par.get("tp"), par.get("pp")) != (before.parallel.dp, before.parallel.tp, before.parallel.pp):
            return self._abort(st, spec, "InvalidAction",
                               f"ApplySettings keeps the parallelism: target {par} != current {before.parallel}")
        if spec.target_gpu_ids and sorted(spec.target_gpu_ids) != sorted(before.gpu_ids):
            return self._abort(st, spec, "InvalidAction", "ApplySettings keeps the GPUs: target.allocation differs")
        gs = [pool.gpus[g] for g in before.gpu_ids if g in pool.gpus]
        cand = pool._cand(gs, "current GPUs") if gs else None
        plan = ExecutionPlan(before.parallel.dp, before.parallel.tp, before.parallel.pp,
                             cand.nodes if cand else [], cand.gpus_per_node if cand else 0,
                             cand.interconnect if cand else "unknown", adapter.name, list(before.gpu_ids),
                             "settings only: parallelism and GPUs kept")
        try:
            settings, previous, why = adapter.plan_settings(spec, plan)
        except AdapterError as e:
            settings, previous, why = None, None, str(e)
        if why:
            return self._abort(st, spec, "SettingsRejected", why)
        if previous and settings["batch"] == previous.get("batch"):
            return self._abort(st, spec, "InvalidAction",
                               f"settings unchanged ({settings['batch']}), nothing to apply")
        st.execution_plan = {"snapshotVersion": pool.version, **plan.to_dict(), "parallelismSource": "current",
                             "allocation": {"source": "current", "gpuIds": list(before.gpu_ids)},
                             "settings": settings, "previousSettings": previous}
        return self._execute(st, spec, adapter, pool, before, plan, deadline)

    def _execute(self, st, spec, adapter, pool, before, plan, deadline):
        cur = len(before.gpu_ids)
        try:
            adapter.prepare(spec, plan)
        except AdapterError as e:
            return self._abort(st, spec, "AdapterRejected", str(e))
        log.info("[%s]   ExecutionPlan: %s on %s (%s) — %s", spec.name, plan.parallel,
                 plan.gpu_ids, plan.interconnect, plan.reason)
        while True:
            ok, detail = adapter.preparation_ready(spec, plan)
            if ok:
                break
            if time.monotonic() > deadline:
                return self._abort(st, spec, "PreparationTimeout",
                                   f"preparation not completed within {spec.safe_policy.timeout_seconds:.0f}s: {detail}")
            time.sleep(self.poll)
        self._enter(st, spec, Phase.PREPARING, "preparation=Ready",
                    f"plan {before.parallel} → {plan.parallel} on {plan.gpu_ids}")

        if spec.safe_policy.mode == "Immediate":
            self._enter(st, spec, Phase.CHECKPOINT, "checkpoint=Skipped", "safePolicy=Immediate")
        else:
            self._enter(st, spec, Phase.CHECKPOINT, "checkpoint=Requested", "waiting for Safe Point")
            adapter.request_safe_point(spec)
            t0 = time.monotonic()
            while not adapter.safe_point_ready(spec):
                if time.monotonic() > deadline:
                    adapter.cancel_safe_point(spec)
                    return self._abort(st, spec, "PreparationTimeout",
                                       f"Safe Point not reached within {spec.safe_policy.timeout_seconds:.0f}s, "
                                       "keeping previous configuration")
                time.sleep(self.poll)
            st.safe_point_ready = True
            ok, detail = adapter.verify_checkpoint(spec)
            if not ok:
                adapter.cancel_safe_point(spec)
                return self._abort(st, spec, "CheckpointInvalid", detail)
            self._enter(st, spec, Phase.CHECKPOINT, detail,
                        f"Safe Point reached in {time.monotonic() - t0:.2f}s")

        self._enter(st, spec, Phase.RECONFIGURE, "restart=InProgress",
                    f"{before.parallel} / GPU {cur} → {plan.parallel} / GPU {plan.parallel.gpus}")
        try:
            after = adapter.reconfigure(spec, plan)
            self._enter(st, spec, Phase.RECONFIGURE, "restart=Running")
            ok, detail = adapter.verify_ready(spec)
            if not ok:
                raise AdapterError(f"response check after restart failed: {detail}")
        except AdapterError as e:
            try:
                adapter.rollback(spec, before)
                prev, rb = Previous.RESTORED, "restored to previous configuration"
            except AdapterError as e2:
                prev, rb = Previous.LOST, f"restore failed: {e2}"
            adapter.finalize(spec)
            return self._abort(st, spec, "ReconfigureFailed", f"{e} / {rb}", prev)
        st.after = after.to_dict()

        released = [g for g in before.gpu_ids if g not in plan.gpu_ids]
        added = [g for g in plan.gpu_ids if g not in before.gpu_ids]
        pool.release(spec.workload_ref.key, released)
        pool.assign(spec.workload_ref.key, added)
        st.released_gpus = released
        adapter.finalize(spec)
        self._enter(st, spec, Phase.SUCCEEDED, detail,
                    f"{after.parallel}, released GPUs {released or '-'}, added GPUs {added or '-'}")
        return st
