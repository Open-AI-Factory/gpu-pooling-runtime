from pathlib import Path

from grp_runtime.cli import load
from grp_runtime.models import RuntimeActionSpec
from grp_runtime.pool import GrpPoolStatusClient
from grp_runtime.runtime_adapter.mock import MockAdapter
from grp_runtime.runtime_coordination.runtime_action_controller.controller import RuntimeActionController, validate
from test_k8s_logic import k8s_adapter, pool_doc
from test_profiles import MIG, launch

EX = Path(__file__).resolve().parents[1] / "scenarios" / "runtime-action"


def mock_spec(action, **target):
    spec = load(str(EX / "ra-scaledown-mock.yaml"))
    spec.action, spec.target_gpu_count = action, 0
    spec.adapter_options = {**spec.adapter_options, "safePointAfterSeconds": 0.02}
    spec.target_settings = target.get("settings")
    return spec


def run(spec, tmp_path):
    return RuntimeActionController(tmp_path, poll_interval=0.02).run(spec)


def test_observe_records_state_without_touching_the_workload(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(MockAdapter, "reconfigure", lambda self, spec, plan: calls.append(plan))
    monkeypatch.setattr(MockAdapter, "observe", lambda self, spec: {"settingsMatch": True, "step": 7})
    st = run(mock_spec("Observe"), tmp_path)
    assert st.phase.value == "Succeeded" and st.result == "observed settings=Applied"
    assert [c["phase"] for c in st.conditions] == ["Accepted", "Preparing", "Succeeded"]
    assert st.observation["step"] == 7 and "observedAt" in st.observation
    assert st.to_dict()["observation"]["settingsMatch"] is True and not calls


def test_apply_settings_needs_an_adapter_that_can_plan_settings(tmp_path):
    st = run(mock_spec("ApplySettings", settings={"maxMicroBatch": 64}), tmp_path)
    assert st.phase.value == "Aborted" and st.reason == "SettingsRejected" and st.previous == "Kept"


def test_apply_settings_keeps_layout_and_runs_checkpoint_restart(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(MockAdapter, "plan_settings", lambda self, spec, plan: (
        seen.append(plan) or {"batch": {"microBatch": 64, "gradientAccumulation": 4}},
        {"batch": {"microBatch": 256, "gradientAccumulation": 1}}, ""))
    st = run(mock_spec("ApplySettings", settings={"maxMicroBatch": 64}), tmp_path)
    assert st.phase.value == "Succeeded"
    assert [c["phase"] for c in st.conditions][-4:] == ["Checkpoint", "Reconfigure", "Reconfigure", "Succeeded"]
    assert (st.after["dp"], st.after["tp"], st.after["pp"]) == (st.before["dp"], st.before["tp"], st.before["pp"])
    assert st.after["gpuIds"] == st.before["gpuIds"] and st.released_gpus == []
    assert st.execution_plan["settings"]["batch"]["microBatch"] == 64
    assert st.execution_plan["previousSettings"]["batch"]["microBatch"] == 256
    assert seen[0].gpu_ids == st.before["gpuIds"]


def test_apply_settings_without_change_or_with_new_layout_is_refused(tmp_path, monkeypatch):
    same = {"batch": {"microBatch": 256, "gradientAccumulation": 1}}
    monkeypatch.setattr(MockAdapter, "plan_settings", lambda self, spec, plan: (same, same, ""))
    st = run(mock_spec("ApplySettings", settings={}), tmp_path)
    assert st.reason == "InvalidAction" and "unchanged" in st.message
    spec = mock_spec("ApplySettings", settings={})
    spec.target_gpu_count = 2
    assert run(spec, tmp_path).reason == "InvalidAction"


def test_validation_of_new_actions():
    base = {"metadata": {"name": "ra"}, "spec": {"workloadRef": "w", "runtime": "ddp", "poolSnapshot": "x"}}
    ok = RuntimeActionSpec.from_manifest({**base, "spec": {**base["spec"], "action": "Observe", "target": {}}})
    assert validate(ok) == []
    bad = RuntimeActionSpec.from_manifest({**base, "spec": {**base["spec"], "action": "Observe",
                                                            "target": {"settings": {"maxMicroBatch": 1}}}})
    assert any("Observe changes nothing" in e for e in validate(bad))
    misuse = RuntimeActionSpec.from_manifest({**base, "spec": {**base["spec"], "action": "Resize",
                                                               "target": {"gpus": 2, "settings": {}}}})
    assert any("only used by ApplySettings" in e for e in validate(misuse))


def k8s_with_profiles():
    from grp_runtime.models import ParallelConfig
    doc = pool_doc([f"default/w-g1-r{i}" for i in range(4)])
    for r in doc["status"]["resources"]:
        r["model"] = MIG
    pool = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool", doc)
    settings = launch()["settings"]
    ep = {"metadata": {"name": "w", "uid": "u"}, "spec": {"runtime": "pytorch", "launch": {"gpu": 1, "settings": settings},
          "parallelism": {"dp": 4, "tp": 1, "pp": 1},
          "planning": {"selected": {"dp": 4, "tp": 1, "pp": 1},
                       "settings": {"effectiveBatch": 1024, "batch": {"microBatch": 128, "gradientAccumulation": 2}}}},
          "status": {"generation": 1}}
    return pool, ep, k8s_adapter(pool, ep), ParallelConfig(4, 1, 1)


def test_k8s_plan_settings_reselects_with_target_override():
    from grp_runtime.models import ExecutionPlan
    pool, ep, a, par = k8s_with_profiles()
    ids = [g.id for g in pool.gpus_of("default/w")]
    plan = ExecutionPlan(4, 1, 1, ["gpu-node-a"], 4, "MIG", "k8s", ids, "settings only")
    spec = RuntimeActionSpec.from_manifest({"metadata": {"name": "ra"}, "spec": {
        "workloadRef": "w", "action": "ApplySettings", "target": {"settings": {"maxMicroBatch": 64}}}})
    new, previous, why = a.plan_settings(spec, plan)
    assert new is None and why.startswith("noFeasiblePlan") and previous["batch"]["microBatch"] == 128
    spec.target_settings = {"maxMicroBatch": 64, "profiles": []}
    new, _, why = a.plan_settings(spec, plan)
    assert why == "" and new["source"] == "rule" and new["batch"] == {"microBatch": 64, "gradientAccumulation": 4}
    spec.target_settings = None
    new, _, _ = a.plan_settings(spec, plan)
    assert new["source"] == "profile" and new["profiles"]["profileId"] == "dp4-128"


def test_k8s_relaunch_with_settings_patches_launch_and_can_be_reverted():
    pool, ep, a, par = k8s_with_profiles()
    ids = [g.id for g in pool.gpus_of("default/w")]
    spec = RuntimeActionSpec.from_manifest({"metadata": {"name": "ra"}, "spec": {"workloadRef": "w"}})
    old = ep["spec"]["launch"]["settings"]
    new = {"effectiveBatch": 1024, "maxMicroBatch": 64}
    a._relaunch(spec, par, ids, "/ckpt", "ra", "apply execution settings", new)
    patch = a.patches[-1][0]
    assert patch["launch"]["settings"]["maxMicroBatch"] == 64 and patch["launch"]["settings"]["profiles"] is None
    assert patch["planning"]["settings"]["batch"] == {"microBatch": 64, "gradientAccumulation": 4}
    assert a.wl.calls[-1][5]["settings"]["source"] == "rule"
    a._relaunch(spec, par, ids, "/ckpt", "ra-rollback", "rollback to previous", old)
    assert a.patches[-1][0]["launch"]["settings"]["profiles"] == old["profiles"]
    assert a.patches[-1][0]["planning"]["settings"]["profiles"]["profileId"] == "dp4-128"


def test_replacing_patch_drops_keys_of_the_previous_settings():
    from grp_runtime.runtime_adapter.k8s import replacing
    old = {"source": "profile", "batch": {"microBatch": 128}, "profiles": {"profileId": "p"}, "memoryBudgetBytesPerDevice": 1}
    new = {"source": "rule", "batch": {"microBatch": 256}}
    assert replacing(old, new) == {"source": "rule", "batch": {"microBatch": 256}, "profiles": None,
                                   "memoryBudgetBytesPerDevice": None}
    assert replacing(None, new) == new
