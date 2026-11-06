import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from grp_runtime.execution_plan_controller import intake
from grp_runtime.execution_plan_controller.execution_plan import validate
from grp_runtime.pool import MockPoolClient

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "scenarios" / "plan-request"


def req(name: str) -> dict:
    return yaml.safe_load((REQ / name).read_text(encoding="utf-8"))


class FakeKubectl:
    def __init__(self, items=(), list_rc=0, create_rc=0, create_err=""):
        self.items, self.list_rc, self.create_rc, self.create_err = list(items), list_rc, create_rc, create_err
        self.calls = []

    def __call__(self, args, input=None, capture_output=True, text=True):
        self.calls.append((args, input))
        if args[1] == "get":
            return subprocess.CompletedProcess(args, self.list_rc, json.dumps({"items": self.items}),
                                               "connection refused" if self.list_rc else "")
        return subprocess.CompletedProcess(args, self.create_rc, "created", self.create_err)


def test_selected_plan_from_request():
    res = intake.generate(req("pr-train-kind.yaml"))
    assert res.result == "Generated"
    d = res.decision
    assert d["mode"] == "Selected" and d["selected"] == {"dp": 2, "tp": 1, "pp": 1}
    assert d["workers"]["nodes"] == 2 and d["workers"]["gpusPerNode"] == 1 and d["workers"]["world"] == 2
    assert [x["nodeSlot"] for x in d["workers"]["rankSlots"]] == [0, 1]
    assert "constraint" not in d and "pinnedNodes" not in res.manifest["spec"]["planning"]
    rows = {(r["dp"], r["tp"], r["pp"]): r for r in d["candidates"]}
    assert set(rows) == {(2, 1, 1), (1, 2, 1), (1, 1, 2)}
    assert "TP=2 exceeds GPUs per node" in rows[(1, 2, 1)]["note"]
    assert "PP>1" in rows[(1, 1, 2)]["note"]
    spec = res.manifest["spec"]
    assert spec["planning"]["requestId"] == "req-train-0001" and spec["planning"]["poolVersion"] == "kind-v1"
    assert spec["planning"]["gpuCount"] == {"value": 2, "source": "request", "analyzed": False}
    assert spec["planning"]["selectionRule"].startswith("rule-based")
    assert res.manifest["metadata"]["labels"][intake.LABEL_REQUEST] == "req-train-0001"


def test_generated_manifest_passes_existing_plan_controller_validation():
    res = intake.generate(req("pr-train-kind.yaml"))
    out = validate(res.manifest, MockPoolClient(ROOT / "mock" / "kind-snapshot.yaml"))
    assert out["validation"] == "Accepted", out["reason"]
    assert out["placement"]["by"] == "scheduler-mock" and out["placement"]["state"] == "Placed"
    assert out["assignedNodes"] == ["grp-worker", "grp-worker2"]
    assert out["placement"]["rankNodes"] == ["grp-worker", "grp-worker2"]


def test_gpus_per_node_derived_from_topology_when_not_given():
    doc = req("pr-train-kind.yaml")
    doc["spec"].update(poolSnapshot="mock/cluster-snapshot.yaml", runtime="ddp", profiling={},
                       resources={"gpus": 2})
    res = intake.generate(doc)
    w = res.decision["workers"]
    assert res.result == "Generated" and w["gpusPerNodeSource"] == "topology"
    assert (w["nodes"], w["gpusPerNode"]) == (1, 2)
    assert res.decision["topology"]["usableGpusPerNode"] == {"node-a": 2, "node-b": 2}


def test_manifest_keeps_existing_spec_fields():
    hand = yaml.safe_load((ROOT / "scenarios" / "execution-plan" / "ep-train-plan.yaml").read_text(encoding="utf-8"))
    gen = intake.generate(req("pr-train-kind.yaml")).manifest
    assert set(hand["spec"]) <= set(gen["spec"])
    for k in ("workloadRef", "runtime", "parallelism", "placement", "profiling", "poolSnapshot", "launch"):
        assert gen["spec"][k] == hand["spec"][k], k


def test_given_parallelism_is_kept_and_placed():
    res = intake.generate(req("pr-train-fixed.yaml"))
    assert res.result == "Generated" and res.decision["mode"] == "Given"
    assert res.decision["selected"] == {"dp": 2, "tp": 1, "pp": 1}
    assert res.decision["execution"]["executable"] is True and "launch" in res.manifest["spec"]
    assert [x["gpu"] for x in res.decision["rankMap"]] == ["grp-worker:0", "grp-worker2:0"]
    out = validate(res.manifest, MockPoolClient(ROOT / "mock" / "kind-snapshot.yaml"))
    assert out["placement"]["state"] == "Placed"
    assert out["placement"]["rankNodes"] == ["grp-worker", "grp-worker2"]


def test_given_3d_parallelism_placement_candidates():
    res = intake.generate(req("pr-given-3d.yaml"))
    d = res.decision
    assert res.result == "Generated" and d["selected"] == {"dp": 2, "tp": 2, "pp": 2}
    assert d["gpuCount"] == {"value": 8, "source": "parallelism", "analyzed": False}
    cands = d["placementCandidates"]
    assert len(cands) >= 2 and cands[0]["selected"] and not any(c["selected"] for c in cands[1:])
    best = cands[0]
    assert best["tpGroups"] == "intra-node" and best["ppStages"] == "one stage per node"
    assert best["layout"]["gpu-node-a"] == ["D1P1T1", "D1P1T2", "D2P1T1", "D2P1T2"]
    split = next(c for c in cands if c["tpGroups"] == "cross-node")
    assert split["ppStages"] == "mixed per node" and split["tpCrossNode"] > 0
    nodes = {x["rank"]: x["node"] for x in d["rankMap"]}
    for dp_ in range(2):
        for pp_ in range(2):
            group = [x for x in d["rankMap"] if x["dp"] == dp_ and x["pp"] == pp_]
            assert len({x["node"] for x in group}) == 1
    assert len(set(nodes.values())) == 2


def test_given_3d_parallelism_is_plan_only_on_k8s_adapter():
    res = intake.generate(req("pr-given-3d.yaml"))
    assert res.decision["execution"]["executable"] is False
    spec = res.manifest["spec"]
    assert spec["planning"]["planOnly"] is True and "launch" not in spec
    assert spec["planning"]["pinnedNodes"] == ["gpu-node-a", "gpu-node-b"] and len(spec["planning"]["rankMap"]) == 8
    out = validate(res.manifest, MockPoolClient(ROOT / "mock" / "two-node-nvlink-snapshot.yaml"))
    assert out["validation"] == "Accepted" and "plan only" in out["reason"]
    assert out["placement"]["state"] == "Placed"
    assert out["placement"]["rankNodes"] == [x["node"] for x in spec["planning"]["rankMap"]]


def test_given_parallelism_without_enough_gpus():
    doc = req("pr-given-3d.yaml")
    doc["spec"]["parallelism"] = {"dp": 4, "tp": 2, "pp": 2}
    res = intake.generate(doc)
    assert res.reason == "InsufficientResources" and "16 GPU(s)" in res.message


def test_invalid_request_lists_field_errors():
    res = intake.generate(req("pr-invalid.yaml"))
    assert res.result == "Rejected" and res.reason == "InvalidSpec"
    fields = {e["field"]: e["code"] for e in res.errors}
    assert fields["spec.runtime"] == "Required"
    assert fields["spec.resources.gpus"] == "OutOfRange"
    assert fields["spec.parallelism.dp"] == "InvalidType"
    assert res.manifest is None


@pytest.mark.parametrize("mutate, code", [
    (lambda s: s.pop("requestId"), "Required"),
    (lambda s: s.update(requestId="has space"), "InvalidValue"),
    (lambda s: s.update(resources={"gpus": 3, "gpusPerNode": 2}), "Inconsistent"),
    (lambda s: s.update(parallelism={"dp": 1, "tp": 1, "pp": 1}), "Inconsistent"),
    (lambda s: s.update(poolSnapshot="mock/none.yaml"), "NotFound"),
    (lambda s: s["profiling"].update(weightsGb=-1), "InvalidValue"),
    (lambda s: s.update(runtime="tensorflow"), "InvalidValue"),
])
def test_invalid_fields(mutate, code):
    doc = req("pr-train-kind.yaml")
    mutate(doc["spec"])
    res = intake.generate(doc)
    assert res.reason == "InvalidSpec" and code in {e["code"] for e in res.errors}


def test_tp_not_supported_by_adapter_is_plan_only():
    res = intake.generate(req("pr-tp-unsupported.yaml"))
    assert res.result == "Generated" and res.decision["execution"]["executable"] is False
    assert "TP>1" in res.decision["execution"]["reason"]


def test_runtime_not_supported_by_k8s_adapter():
    doc = req("pr-train-kind.yaml")
    doc["spec"]["runtime"] = "deepspeed"
    assert intake.generate(doc).reason == "AdapterRejected"


def test_insufficient_resources():
    res = intake.generate(req("pr-insufficient.yaml"))
    assert res.reason == "InsufficientResources" and "cannot host 4 node(s) x 1 GPU" in res.message


def test_no_feasible_plan_records_all_candidates():
    doc = req("pr-train-kind.yaml")
    doc["spec"]["profiling"]["weightsGb"] = 10
    res = intake.generate(doc)
    assert res.reason == "NoFeasiblePlan"
    assert len(res.decision["candidates"]) == 3 and not any(r["feasible"] for r in res.decision["candidates"])


def test_given_parallelism_infeasible_memory():
    doc = req("pr-train-fixed.yaml")
    doc["spec"]["profiling"]["weightsGb"] = 10
    res = intake.generate(doc)
    assert res.reason == "NoFeasiblePlan" and res.decision["mode"] == "Given"


def test_pool_version_mismatch():
    res = intake.generate(req("pr-pool-mismatch.yaml"))
    assert res.reason == "PoolSnapshotMismatch" and "kind-v0" in res.message


def test_same_request_returns_previous(tmp_path):
    doc = req("pr-train-kind.yaml")
    intake.record(intake.generate(doc), tmp_path)
    prev = intake.previous(doc, tmp_path)
    assert prev.result == "Generated" and prev.manifest["metadata"]["name"] == "train-plan"


def test_same_request_id_different_content_rejected(tmp_path):
    doc = req("pr-train-kind.yaml")
    intake.record(intake.generate(doc), tmp_path)
    changed = copy.deepcopy(doc)
    changed["spec"]["resources"]["gpus"] = 1
    prev = intake.previous(changed, tmp_path)
    assert prev.result == "Rejected" and prev.reason == "DuplicateRequest"


def test_rejected_record_does_not_block_retry(tmp_path):
    doc = req("pr-insufficient.yaml")
    intake.record(intake.generate(doc), tmp_path)
    assert intake.previous(doc, tmp_path) is None


def test_register_success_uses_kubectl_create():
    res = intake.generate(req("pr-train-kind.yaml"))
    k = FakeKubectl()
    out = intake.register(res, runner=k)
    assert out.result == "Registered"
    args, body = k.calls[-1]
    assert args[:3] == ["kubectl", "create", "-f"] and yaml.safe_load(body) == res.manifest


def test_register_duplicate_request_id():
    res = intake.generate(req("pr-train-kind.yaml"))
    k = FakeKubectl(items=[{"metadata": {"name": "train-plan", "labels": {intake.LABEL_REQUEST: "req-train-0001"}},
                            "spec": {"workloadRef": "train"}}])
    out = intake.register(res, runner=k)
    assert out.result == "Duplicate" and len(k.calls) == 1


def test_register_conflict_with_existing_plan_for_workload():
    res = intake.generate(req("pr-train-kind.yaml"))
    k = FakeKubectl(items=[{"metadata": {"name": "train-plan"}, "spec": {"workloadRef": "train"}}])
    out = intake.register(res, runner=k)
    assert out.result == "Rejected" and out.reason == "DuplicateRequest" and len(k.calls) == 1


def test_register_cluster_unreachable():
    res = intake.generate(req("pr-train-kind.yaml"))
    out = intake.register(res, runner=FakeKubectl(list_rc=1))
    assert out.reason == "RegistrationFailed" and "connection refused" in out.message


def test_register_create_failure():
    res = intake.generate(req("pr-train-kind.yaml"))
    out = intake.register(res, runner=FakeKubectl(create_rc=1, create_err='admission webhook denied'))
    assert out.reason == "RegistrationFailed" and "denied" in out.message


def test_plan_has_no_node_names_scheduler_places_after():
    mf = intake.generate(req("pr-train-kind.yaml")).manifest
    assert mf["spec"]["placement"] == {"nodes": 2, "gpusPerNode": 1}
    assert "pinnedNodes" not in mf["spec"]["planning"]


def test_unschedulable_when_pool_shrinks_after_plan(tmp_path):
    (tmp_path / "mock").mkdir()
    snap = tmp_path / "mock" / "kind-snapshot.yaml"
    shutil.copy(ROOT / "mock" / "kind-snapshot.yaml", snap)
    res = intake.generate(req("pr-train-kind.yaml"), root=tmp_path)
    doc = yaml.safe_load(snap.read_text(encoding="utf-8"))
    doc["nodes"][1]["gpus"][0]["allocatedTo"] = "other/workload"
    snap.write_text(yaml.safe_dump(doc), encoding="utf-8")
    out = validate(res.manifest, MockPoolClient(snap))
    assert out["validation"] == "Accepted" and out["placement"]["state"] == "Unschedulable"
    assert out["assignedNodes"] == [] and "cannot secure 2 node(s)" in out["placement"]["reason"]


def test_pinned_gpus_constrain_decision_and_placement():
    res = intake.generate(req("pr-train-allocated.yaml"))
    assert res.result == "Generated" and res.decision["constraint"]["nodes"] == ["grp-worker2"]
    assert res.decision["workers"]["gpusPerNodeSource"] == "pinned"
    assert res.decision["selected"] == {"dp": 1, "tp": 1, "pp": 1}
    assert res.manifest["spec"]["planning"]["pinnedNodes"] == ["grp-worker2"]
    out = validate(res.manifest, MockPoolClient(ROOT / "mock" / "kind-snapshot.yaml"))
    assert out["placement"]["state"] == "Placed" and out["assignedNodes"] == ["grp-worker2"]


@pytest.mark.parametrize("alloc, code", [
    ({"gpuIds": ["grp-worker:0"]}, "Inconsistent"),
    ({"gpuIds": ["grp-worker:0", "grp-worker:0"]}, "InvalidValue"),
    ({"gpuIds": "grp-worker:0"}, "InvalidType"),
])
def test_allocation_field_errors(alloc, code):
    doc = req("pr-train-kind.yaml")
    doc["spec"]["allocation"] = alloc
    res = intake.generate(doc)
    assert res.reason == "InvalidSpec" and code in {e["code"] for e in res.errors}


def test_allocation_in_use_by_other_workload():
    doc = req("pr-train-kind.yaml")
    doc["spec"].update(poolSnapshot="mock/cluster-snapshot.yaml", runtime="ddp", profiling={},
                       allocation={"gpuIds": ["node-a:0", "node-a:4"]})
    assert intake.generate(doc).reason == "InsufficientResources"


def test_uneven_pinned_gpus_return_feedback():
    doc = req("pr-train-kind.yaml")
    doc["spec"].update(poolSnapshot="mock/cluster-snapshot.yaml", runtime="ddp", profiling={},
                       resources={"gpus": 3}, allocation={"gpuIds": ["node-a:4", "node-a:5", "node-b:0"]})
    res = intake.generate(doc)
    assert res.reason == "NoFeasiblePlan" and "uneven" in res.message
    assert res.decision["feedback"]["options"]


def test_memory_infeasible_feedback_names_required_gpu_memory():
    doc = req("pr-train-kind.yaml")
    doc["spec"]["profiling"]["weightsGb"] = 10
    res = intake.generate(doc)
    opt = res.decision["feedback"]["options"][0]
    assert res.reason == "NoFeasiblePlan" and opt["minGpuMemoryGb"] > 2 and (opt["tp"], opt["pp"]) == (1, 1)


def test_pinned_nodes_without_capacity_unschedulable():
    res = intake.generate(req("pr-train-allocated.yaml"))
    mf = copy.deepcopy(res.manifest)
    mf["spec"]["planning"]["pinnedNodes"] = ["no-such-node"]
    out = validate(mf, MockPoolClient(ROOT / "mock" / "kind-snapshot.yaml"))
    assert out["placement"]["state"] == "Unschedulable" and "lack" in out["placement"]["reason"]


def test_spec_update_after_reconfigure_validates_without_placement():
    mf = copy.deepcopy(intake.generate(req("pr-train-allocated.yaml")).manifest)
    mf["spec"].update(parallelism={"dp": 2, "tp": 1, "pp": 1}, placement={"nodes": 2, "gpusPerNode": 1})
    mf["status"] = {"generation": 2}
    out = validate(mf, MockPoolClient(ROOT / "mock" / "kind-snapshot.yaml"), place_workers=False)
    assert out["validation"] == "Accepted", out["reason"]
    assert "placement" not in out and "assignedNodes" not in out


class FakeClient:
    def __init__(self, items=()):
        self.items, self.created = list(items), []

    def list_plans(self, namespace):
        return self.items

    def create_plan(self, manifest):
        self.created.append(manifest)


def resource(name: str) -> dict:
    body = req(name)
    body["metadata"].update(namespace="default", uid="uid-123")
    body["status"] = {}
    return body


def test_resource_watch_generates_and_registers_with_ref():
    c = FakeClient()
    res = intake.from_resource(resource("pr-train-kind.yaml"), c)
    assert res.result == "Registered" and len(c.created) == 1
    ref = c.created[0]["spec"]["planning"]["requestRef"]
    assert ref == {"kind": "PlanRequest", "name": "train-plan", "namespace": "default", "uid": "uid-123"}
    st = intake.status_summary(res)
    assert st["executionPlan"] == "train-plan" and st["workers"]["nodes"] == 2 and "constraint" not in st
    assert "placement" not in st
    assert st["selected"] == {"dp": 2, "tp": 1, "pp": 1} and len(st["candidates"]) == 3


def test_resource_watch_rejection_is_summarized():
    c = FakeClient()
    res = intake.from_resource(resource("pr-invalid.yaml"), c)
    st = intake.status_summary(res)
    assert st["result"] == "Rejected" and st["reason"] == "InvalidSpec" and st["errors"] and not c.created


def test_resource_watch_duplicate_request():
    first = FakeClient()
    intake.from_resource(resource("pr-train-kind.yaml"), first)
    again = FakeClient(items=first.created)
    body = resource("pr-train-kind.yaml")
    body["metadata"]["name"] = "train-plan-resend"
    res = intake.from_resource(body, again)
    assert res.result == "Duplicate" and not again.created
    assert intake.status_summary(res)["executionPlan"] == "train-plan"


def test_register_detects_pool_change_after_decision(tmp_path):
    (tmp_path / "mock").mkdir()
    snap = tmp_path / "mock" / "kind-snapshot.yaml"
    shutil.copy(ROOT / "mock" / "kind-snapshot.yaml", snap)
    res = intake.generate(req("pr-train-kind.yaml"), root=tmp_path)
    assert res.result == "Generated"
    snap.write_text(snap.read_text(encoding="utf-8").replace("kind-v1", "kind-v2"), encoding="utf-8")
    k = FakeKubectl()
    out = intake.register(res, root=tmp_path, runner=k)
    assert out.reason == "PoolSnapshotMismatch" and all(c[0][1] != "create" for c in k.calls)


def test_plan_owned_by_request_resource():
    ref = {"kind": "PlanRequest", "name": "train-plan", "namespace": "default", "uid": "u-1"}
    md = intake.generate(req("pr-train-kind.yaml"), request_ref=ref).manifest["metadata"]
    assert md["ownerReferences"] == [{"apiVersion": "grp.keti.re.kr/v1alpha1", "kind": "PlanRequest",
                                      "name": "train-plan", "uid": "u-1"}]
    assert "ownerReferences" not in intake.generate(req("pr-train-kind.yaml")).manifest["metadata"]


def test_live_pool_reference_is_accepted_and_unavailable_pool_is_rejected(monkeypatch):
    doc = req("pr-train-kind.yaml")
    doc["spec"]["poolSnapshot"] = "k8s:gpuresourcepools/gpu-pool"
    assert intake.parse_request(doc)[1] == []

    def boom(name):
        raise RuntimeError("not found")
    monkeypatch.setattr("grp_runtime.pool.fetch_pool", boom)
    res = intake.generate(doc)
    assert (res.result, res.reason) == ("Rejected", "PoolUnavailable")


def test_given_plan_carries_process_groups_and_communication():
    res = intake.generate(req("pr-given-3d.yaml"))
    d = res.decision
    coord = {x["rank"]: x for x in d["rankMap"]}
    groups = d["processGroups"]
    for kind, fixed in (("tp", ("dp", "pp")), ("dp", ("tp", "pp")), ("pp", ("dp", "tp"))):
        assert sorted(r for g in groups[kind] for r in g) == list(range(8))
        for g in groups[kind]:
            assert len({tuple(coord[r][k] for k in fixed) for r in g}) == 1
    assert all(len({coord[r]["node"] for r in g}) == 1 for g in groups["tp"])
    assert d["communication"]["backend"] == "nccl"
    planning = res.manifest["spec"]["planning"]
    assert planning["processGroups"] == groups and planning["communication"]["port"] == 29500


def test_allocation_source_marks_planner_fallback():
    fixed = intake.generate(req("pr-train-fixed.yaml")).decision
    assert fixed["allocation"]["source"] == "planner-fallback"
    assert fixed["communication"]["backend"] == "gloo"
    assert intake.generate(req("pr-train-allocated.yaml")).decision["allocation"]["source"] == "request"
