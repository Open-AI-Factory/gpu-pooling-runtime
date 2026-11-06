import json
from pathlib import Path

import pytest
import yaml

from grp_runtime.cli import load
from grp_runtime.runtime_coordination.runtime_action_controller.controller import RuntimeActionController
from grp_runtime.models import Profiling
from grp_runtime.execution_plan_controller.planner import build_plan, enumerate_configs
from grp_runtime.pool import Candidate

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
EX = SCENARIOS / "runtime-action"


def run(name, tmp_path, **opts):
    spec = load(str(EX / name))
    spec.adapter_options.update(opts)
    return RuntimeActionController(tmp_path, poll_interval=0.02).run(spec)


def phases(st):
    return [c["phase"] for c in st.conditions]


def test_enumerate_configs_product():
    for dp, tp, pp in enumerate_configs(8):
        assert dp * tp * pp == 8
    assert len(enumerate_configs(8)) == 10


def test_plan_prefers_tp_on_nvlink_pp_on_pcie():
    prof = Profiling(model_memory_gb=100, activation_memory_gb=10, num_layers=40, num_attention_heads=40)
    nv = Candidate(["a:0", "a:1"], ["a"], "NVLink", 2, 80)
    pc = Candidate(["b:0", "b:1"], ["b"], "PCIe", 2, 80)
    assert (build_plan(2, prof, nv, "x").tp, build_plan(2, prof, nv, "x").pp) == (2, 1)
    assert (build_plan(2, prof, pc, "x").tp, build_plan(2, prof, pc, "x").pp) == (1, 2)


def test_plan_tp_never_crosses_node():
    prof = Profiling(model_memory_gb=100, activation_memory_gb=0)
    cand = Candidate(["a:0", "b:0"], ["a", "b"], "Network", 1, 80)
    plan = build_plan(2, prof, cand, "x")
    assert plan.tp == 1 and plan.pp == 2


def test_scaledown_succeeds(tmp_path):
    st = run("ra-scaledown-mock.yaml", tmp_path, safePointAfterSeconds=0.05)
    assert [c["result"] for c in st.conditions] == [
        "request validated", "preparation=InProgress", "preparation=Ready", "checkpoint=Requested",
        "serviceConfig=Valid", "restart=InProgress", "restart=Running", "ready=true response=OK"]
    assert phases(st) == ["Accepted", "Preparing", "Preparing", "Checkpoint", "Checkpoint",
                          "Reconfigure", "Reconfigure", "Succeeded"]
    assert st.before["gpuCount"] == 4 and st.after["gpuCount"] == 2
    assert (st.after["dp"], st.after["tp"], st.after["pp"]) == (1, 2, 1)
    assert st.released_gpus == ["node-a:2", "node-a:3"]


def test_scaledown_follows_target_parallelism_and_allocation(tmp_path):
    spec = load(str(EX / "ra-scaledown-mock.yaml"))
    spec.adapter_options.update(safePointAfterSeconds=0.05)
    spec.target_parallelism, spec.target_gpu_ids = {"dp": 1, "tp": 1, "pp": 2}, ["node-a:1", "node-a:3"]
    st = RuntimeActionController(tmp_path, poll_interval=0.02).run(spec)
    assert st.phase.value == "Succeeded", st.message
    assert (st.after["dp"], st.after["tp"], st.after["pp"]) == (1, 1, 2)
    assert st.after["gpuIds"] == ["node-a:1", "node-a:3"]
    assert st.execution_plan["allocation"]["source"] == "request"
    assert st.execution_plan["parallelismSource"] == "target"
    assert run("ra-scaledown-mock.yaml", tmp_path, safePointAfterSeconds=0.05).execution_plan["allocation"]["source"]         == "planner-fallback"


def test_target_allocation_must_be_free_and_match_target(tmp_path):
    spec = load(str(EX / "ra-scaledown-mock.yaml"))
    spec.target_gpu_ids = ["node-a:1"]
    assert RuntimeActionController(tmp_path).run(spec).reason == "InvalidSpec"
    spec.target_gpu_ids, spec.target_parallelism = ["node-a:1", "node-a:3"], {"dp": 4, "tp": 1, "pp": 1}
    assert RuntimeActionController(tmp_path).run(spec).reason == "InvalidSpec"
    spec.target_parallelism, spec.target_gpu_ids = None, ["node-a:1", "nope:0"]
    st = RuntimeActionController(tmp_path).run(spec)
    assert st.reason == "InsufficientResources" and "nope:0" in st.message


def test_scaleup_adds_same_node_gpus(tmp_path):
    st = run("ra-scaleup-mock.yaml", tmp_path, safePointAfterSeconds=0.05)
    assert st.phase.value == "Succeeded"
    assert st.after["gpuIds"][-2:] == ["node-a:4", "node-a:5"]


@pytest.mark.parametrize("name,reason,previous", [
    ("ra-preparation-timeout-mock.yaml", "PreparationTimeout", "Kept"),
    ("ra-reconfigure-fail-mock.yaml", "ReconfigureFailed", "Restored"),
    ("ra-noplan-mock.yaml", "NoFeasiblePlan", "Kept"),
])
def test_failure_paths(tmp_path, name, reason, previous):
    spec = load(str(EX / name))
    spec.safe_policy.timeout_seconds = min(spec.safe_policy.timeout_seconds, 0.3)
    spec.adapter_options.setdefault("safePointAfterSeconds", 0.05)
    st = RuntimeActionController(tmp_path, poll_interval=0.02).run(spec)
    assert (st.phase.value, st.reason, st.previous) == ("Aborted", reason, previous)
    assert st.result == f"reason={reason} previous={previous}"


def test_preparing_timeout_never_reaches_checkpoint(tmp_path):
    spec = load(str(EX / "ra-scaledown-mock.yaml"))
    spec.safe_policy.timeout_seconds = 0.2
    spec.adapter_options["failAt"] = "Preparing"
    st = RuntimeActionController(tmp_path, poll_interval=0.02).run(spec)
    assert st.reason == "PreparationTimeout" and "Checkpoint" not in phases(st)


def test_invalid_spec(tmp_path):
    spec = load(str(EX / "ra-scaledown-mock.yaml"))
    spec.runtime, spec.target_gpu_count = "tensorflow", 0
    st = RuntimeActionController(tmp_path).run(spec)
    assert st.reason == "InvalidSpec" and phases(st) == ["Accepted", "Aborted"]


def test_scaledown_must_reduce(tmp_path):
    spec = load(str(EX / "ra-scaledown-mock.yaml"))
    spec.target_gpu_count = 4
    assert RuntimeActionController(tmp_path).run(spec).reason == "InvalidAction"


def test_scenarios_are_valid_yaml():
    files = list(SCENARIOS.rglob("*.yaml"))
    assert len(files) >= 13
    for p in files:
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert m["kind"] in ("RuntimeAction", "ExecutionPlan", "InferenceService", "RecoveryRequest",
                             "PlanRequest", "ConfigMap"), p
        if m["kind"] == "ConfigMap":
            from grp_runtime.execution_plan_controller.profiles import validate_profiles
            assert validate_profiles(json.loads(m["data"]["profiles.json"])["acceptedProfiles"]), p
