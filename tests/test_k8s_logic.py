import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from grp_runtime.execution_plan_controller.execution_plan import validate
from grp_runtime.runtime_adapter.workload import Workload
from grp_runtime.pool import GrpPoolStatusClient, MockPoolClient

ROOT = Path(__file__).resolve().parents[1]


class FakeCore:
    def __init__(self, data):
        self.data = data

    def read_namespaced_config_map(self, name, ns):
        return SimpleNamespace(data={k: json.dumps(v) for k, v in self.data.items()})


def agent(rank, node, cpu=1.0, mem=1024, world=2, age=0.0, hook="connected"):
    return {"hook": hook, "rank": rank, "world": world, "node": node, "updatedAt": time.time() - age,
            "isolation": {"cpuLimit": cpu, "memoryLimitMb": mem, "device": "cpu"}}


REQ = [{"rank": r, "node": n, "cpuLimit": 1.0, "memoryLimitMb": 1024, "device": "cpu", "world": 2}
       for r, n in enumerate(["w1", "w2"])]


def readiness(agents):
    return Workload("default", "train", core=FakeCore({f"agent-{a['rank']}": a for a in agents})).readiness(REQ)


def test_ready_when_all_connected_and_isolation_matches():
    r = readiness([agent(0, "w1"), agent(1, "w2")])
    assert (r["hook"], r["isolation"], r["state"], r["ranks"]) == ("connected", "requested-match", "ReadyToRun", [0, 1])


def test_isolation_mismatch_blocks_ready():
    r = readiness([agent(0, "w1"), agent(1, "w1", mem=512)])
    assert r["state"] == "NotReady" and r["isolation"] == "mismatch"
    assert any("rank1.node" in m for m in r["mismatches"]) and any("memoryLimitMb" in m for m in r["mismatches"])


def test_stale_heartbeat_is_not_connected():
    r = readiness([agent(0, "w1"), agent(1, "w2", age=30)])
    assert r["state"] == "NotReady" and r["staleRanks"] == [1] and r["hook"] == "partial"


def test_missing_agent_pending():
    r = readiness([agent(0, "w1")])
    assert r["isolation"] == "pending" and r["state"] == "NotReady"


def ep(**spec):
    base = {"workloadRef": "train", "runtime": "fsdp", "parallelism": {"dp": 2, "tp": 1, "pp": 1},
            "placement": {"nodes": 2, "gpusPerNode": 1}, "interconnect": "rdma",
            "profiling": {"weightsGb": 0.02, "gradientsGb": 0.02, "optimizerGb": 0.04,
                          "activationGbPerSample": 0.000035, "globalBatch": 1024}}
    return {"metadata": {"name": "train-plan", "namespace": "default"}, "spec": {**base, **spec}}


def pool():
    return MockPoolClient(ROOT / "mock" / "kind-snapshot.yaml")


def test_plan_accepted_assigns_two_nodes():
    r = validate(ep(), pool())
    assert r["validation"] == "Accepted" and r["adapter"] == "fsdp"
    assert r["assignedNodes"] == ["grp-worker", "grp-worker2"]


def test_plan_rejected_reasons():
    assert "does not support TP>1" in validate(ep(parallelism={"dp": 1, "tp": 2, "pp": 1}), pool())["reason"]
    assert "≠ nodes×gpusPerNode" in validate(ep(parallelism={"dp": 4, "tp": 1, "pp": 1}), pool())["reason"]
    r = validate(ep(parallelism={"dp": 3, "tp": 1, "pp": 1}, placement={"nodes": 3, "gpusPerNode": 1}), pool())
    assert r["validation"] == "Accepted" and r["placement"]["state"] == "Unschedulable"
    assert "cannot secure 3 node(s)" in r["placement"]["reason"] and r["assignedNodes"] == []
    assert "does not support runtime 'vllm'" in validate(ep(runtime="vllm"), pool())["reason"]


def test_fsdp_memory_grows_when_dp_shrinks():
    big = {"weightsGb": 1.0, "gradientsGb": 1.0, "optimizerGb": 2.0}
    assert validate(ep(profiling=big, parallelism={"dp": 2, "tp": 1, "pp": 1}), pool())["validation"] == "Rejected"
    r2 = validate(ep(profiling={"weightsGb": 0.5, "gradientsGb": 0.5, "optimizerGb": 1.0}), pool())
    r1 = validate(ep(profiling={"weightsGb": 0.5, "gradientsGb": 0.5, "optimizerGb": 1.0},
                     parallelism={"dp": 1, "tp": 1, "pp": 1}, placement={"nodes": 1, "gpusPerNode": 1}), pool())
    assert r2["validation"] == "Accepted" and r1["validation"] == "Rejected"


class FakePods:
    def __init__(self, pods):
        self.items = [SimpleNamespace(metadata=SimpleNamespace(labels={"grp.keti.re.kr/generation": str(g)}),
                                      status=SimpleNamespace(phase=ph)) for g, ph in pods]

    def list_namespaced_pod(self, ns, label_selector=None):
        return SimpleNamespace(items=self.items)


def outcome(pods, gen=1):
    return Workload("default", "train", core=FakePods(pods)).outcome(gen)


def test_outcome_succeeded_only_when_every_rank_succeeded():
    assert outcome([(1, "Succeeded"), (1, "Succeeded")]) == "Succeeded"
    assert outcome([(1, "Succeeded"), (1, "Running")]) is None
    assert outcome([]) is None


def test_outcome_failed_on_any_rank_and_ignores_old_generation():
    assert outcome([(1, "Succeeded"), (1, "Failed")]) == "Failed"
    assert outcome([(1, "Failed"), (2, "Running"), (2, "Succeeded")], gen=2) is None
    assert outcome([(1, "Failed"), (2, "Succeeded")], gen=2) == "Succeeded"


def pool_doc(owners):
    res = [{"node": "gpu-node-a", "resource": f"gpu-0-mig-1g24gb-{i}", "deviceType": "mig", "migProfile": "1g.24gb",
            "uuid": f"MIG-{i}", "parentUuid": "GPU-A", "capacity": "24576Mi", "owner": [o] if o else [],
            "assigned": 1 if o else 0, "unassigned": 0 if o else 1, "nodeFlags": []}
           for i, o in enumerate(owners)]
    return {"kind": "GpuResourcePool", "metadata": {"name": "gpu-pool"},
            "status": {"resources": res, "sourceVersions": {"pods": "1"}}}


def test_live_pool_counts_each_mig_instance_as_own_device():
    from grp_runtime.pool import GrpPoolStatusClient
    p = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool", pool_doc([None, None, "default/other-r0", None]))
    assert len(p.gpus) == 4 and p.interconnect == {"gpu-node-a": "MIG"} and not p.excluded
    assert {g.physical for g in p.free_gpus()} == {"MIG-0", "MIG-1", "MIG-3"}
    assert [g.allocated_to for g in p.gpus.values()] == [None, None, "default/other-r0", None]


def test_live_pool_marks_rank_pods_as_their_workload():
    from grp_runtime.pool import GrpPoolStatusClient
    p = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool",
                            pool_doc(["default/w-g3-r0", "default/w-g3-r1", "default/x-g1-r0", None]))
    assert [g.uuid for g in p.gpus_of("default/w")] == ["MIG-0", "MIG-1"]
    assert [g.uuid for g in p.gpus_of("default/x")] == ["MIG-2"] and [g.uuid for g in p.free_gpus()] == ["MIG-3"]


def test_mig_pool_selects_gloo_unless_request_overrides():
    from grp_runtime.execution_plan_controller.execution_plan import communication
    from grp_runtime.pool import GrpPoolStatusClient
    p = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool", pool_doc([None, None]))
    assert communication(p, ["gpu-node-a"], {"gpu": 1})["backend"] == "gloo"
    assert communication(p, ["gpu-node-a"], {"gpu": 1, "env": {"GRP_DIST_BACKEND": "nccl"}})["backend"] == "nccl"


class RecordingCore:
    def __init__(self):
        self.pods = []

    def read_namespaced_config_map(self, name, ns):
        return SimpleNamespace(data={})

    def read_namespaced_service(self, name, ns):
        return None

    def create_namespaced_pod(self, ns, pod):
        self.pods.append(pod)


def test_launch_follows_plan_rank_layout_and_groups():
    import copy
    import yaml
    from grp_runtime.execution_plan_controller import intake
    doc = yaml.safe_load((ROOT / "scenarios" / "plan-request" / "pr-given-3d.yaml").read_text(encoding="utf-8"))
    spec = copy.deepcopy(intake.generate(doc).manifest["spec"])
    planning = spec["planning"]
    nodes = planning["pinnedNodes"]
    swap = {nodes[0]: nodes[1], nodes[1]: nodes[0]}
    for x in planning["rankMap"]:
        x["node"] = swap[x["node"]]
    by_rank = {x["rank"]: x for x in planning["rankMap"]}
    placed, plan = [by_rank[r]["node"] for r in range(8)], planning
    assert placed != [nodes[r // 4] for r in range(8)]
    core = RecordingCore()
    requested = Workload("default", "w", core=core).launch(8, placed, "pytorch-3d", {}, 1, None, "t",
                                                           parallelism={"dp": 2, "tp": 2, "pp": 2}, plan=plan)
    for pod in core.pods:
        env = {e.name: e.value for e in pod.spec.containers[0].env if e.value is not None}
        x = by_rank[int(env["RANK"])]
        assert pod.spec.node_selector["kubernetes.io/hostname"] == x["node"]
        assert (env["GRP_DP_RANK"], env["GRP_TP_RANK"], env["GRP_PP_RANK"]) == (str(x["dp"]), str(x["tp"]), str(x["pp"]))
        assert json.loads(env["GRP_PROCESS_GROUPS"]) == planning["processGroups"]
        assert env["GRP_DIST_BACKEND"] == "nccl"
    assert [r["plannedGpu"] for r in requested] == [by_rank[r]["gpu"] for r in range(8)]


def test_execution_settings_preserve_effective_batch_across_dp():
    from grp_runtime.execution_plan_controller.profiles import select
    launch = {"settings": {"effectiveBatch": 1024, "maxMicroBatch": 256}}
    four, _ = select(launch, {"dp": 4, "tp": 1, "pp": 1})
    two, _ = select(launch, {"dp": 2, "tp": 1, "pp": 1})
    assert four["batch"] == {"microBatch": 256, "gradientAccumulation": 1}
    assert two["batch"] == {"microBatch": 256, "gradientAccumulation": 2}
    assert two["source"] == "rule"
    assert select({"settings": {"effectiveBatch": 1000, "maxMicroBatch": 256}}, {"dp": 3, "tp": 1, "pp": 1})[1]
    assert select({}, {"dp": 2, "tp": 1, "pp": 1}) == (None, "")
    core = RecordingCore()
    Workload("default", "w", core=core).launch(2, ["n", "n"], "ddp", {}, 1, None, "t",
                                               parallelism={"dp": 2, "tp": 1, "pp": 1}, plan={"settings": two})
    for pod in core.pods:
        env = {e.name: e.value for e in pod.spec.containers[0].env if e.value is not None}
        assert json.loads(env["AI_FACTORY_RUNTIME_CONFIG"]) == two


def test_image_mode_mounts_checkpoint_claim_instead_of_host_code():
    def pod(launch):
        core = RecordingCore()
        Workload("grp-runtime", "w", core=core).launch(1, ["n"], "ddp", launch, 1, None, "t")
        return core.pods[0].spec
    legacy = pod({"hostRoot": "/srv/grp"})
    assert [(v.name, v.host_path.path) for v in legacy.volumes] == [("work", "/srv/grp")]
    assert legacy.service_account_name == "grp-agent"
    image = pod({"codeInImage": True, "checkpoint": {"claimName": "grp-checkpoints"}, "serviceAccount": "grp-rank",
                 "image": "grp-runtime-train:0.1"})
    assert [(v.name, v.persistent_volume_claim.claim_name) for v in image.volumes] == [("checkpoints", "grp-checkpoints")]
    assert [(m.name, m.mount_path) for m in image.containers[0].volume_mounts] == [("checkpoints", "/work/runs/k8s")]
    assert image.service_account_name == "grp-rank" and image.containers[0].image == "grp-runtime-train:0.1"


class FakeWorkload:
    def __init__(self):
        self.calls = []

    def teardown(self):
        self.calls.append(("teardown",))

    def launch(self, world, placed, runtime, launch, gen, resume, tag, parallelism=None, owner=None, plan=None):
        self.calls.append(("launch", world, placed, gen, resume, plan))
        return [{"rank": c["rank"], "claim": f"w-g{gen}-r{c['rank']}-gpu", "plannedUuid": c["uuid"]}
                for c in plan["deviceBinding"]["claims"]]


def k8s_adapter(pool, ep):
    from grp_runtime.runtime_adapter.k8s import K8sAdapter
    a = object.__new__(K8sAdapter)
    a.pool, a.ns, a.ep_name, a.wl, a.patches = pool, "default", "w", FakeWorkload(), []
    a.ep = lambda: ep
    a.patch_ep = lambda spec=None, status=None: a.patches.append((spec, status))
    return a


def test_reconfigure_rebuilds_plan_and_claims_for_new_parallelism():
    from grp_runtime.models import ParallelConfig, RuntimeActionSpec
    from grp_runtime.pool import GrpPoolStatusClient
    pool = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool", pool_doc([f"default/w-g1-r{i}" for i in range(4)]))
    ep = {"metadata": {"name": "w", "uid": "u"}, "spec": {"runtime": "pytorch", "launch": {"gpu": 1},
          "parallelism": {"dp": 4, "tp": 1, "pp": 1}, "planning": {"selected": {"dp": 4, "tp": 1, "pp": 1}}},
          "status": {"generation": 1}}
    a = k8s_adapter(pool, ep)
    spec = RuntimeActionSpec.from_manifest({"metadata": {"name": "ra"}, "spec": {"workloadRef": "w", "target": {"gpus": 2}}})
    keep = [g.id for g in pool.gpus_of("default/w")][1::2]
    assert a._relaunch(spec, ParallelConfig(2, 1, 1), keep, "/ckpt", "ra", "reconfigure 4→2 ranks") == 2
    assert a.wl.calls[0] == ("teardown",)
    _, world, placed, gen, resume, plan = a.wl.calls[1]
    assert (world, placed, gen, resume) == (2, ["gpu-node-a", "gpu-node-a"], 2, "/ckpt")
    assert [c["uuid"] for c in plan["deviceBinding"]["claims"]] == ["MIG-1", "MIG-3"]
    assert plan["processGroups"]["dp"] == [[0, 1]] and plan["communication"]["backend"] == "gloo"
    spec_patch, status = a.patches[0]
    assert spec_patch["parallelism"] == {"dp": 2, "tp": 1, "pp": 1}
    assert spec_patch["planning"]["rankSource"] == "replan" and spec_patch["planning"]["selected"]["dp"] == 2
    assert status["generation"] == 2 and status["history"][-1]["gpuIds"] == keep
    ep["spec"]["launch"]["settings"] = {"effectiveBatch": 1024, "maxMicroBatch": 512}
    a._relaunch(spec, ParallelConfig(2, 1, 1), keep, "/ckpt", "ra", "again")
    assert a.wl.calls[-1][5]["settings"]["batch"] == {"microBatch": 512, "gradientAccumulation": 1}
    assert a.patches[-1][0]["planning"]["settings"]["effectiveBatch"] == 1024
    other = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool", pool_doc(["default/w-g1-r0", "default/x-g1-r0"]))
    from grp_runtime.runtime_adapter.base import AdapterError
    with pytest.raises(AdapterError, match="cannot rebuild execution plan"):
        k8s_adapter(other, ep)._relaunch(spec, ParallelConfig(2, 1, 1), list(other.gpus), "/ckpt", "ra", "x")


def dra_pool(tmp_path):
    res = [{"node": "gpu-node-a", "resource": f"gpu-0-mig-1g24gb-{i}", "deviceType": "mig", "migProfile": "1g.24gb",
            "placement": 3 * i, "uuid": f"MIG-{i}", "parentUuid": "GPU-P", "capacity": "24576Mi", "owner": [],
            "assigned": 0, "unassigned": 1, "nodeFlags": []} for i in range(4)]
    res += [{"node": "gpu-node", "resource": f"gpu-{i}", "deviceType": "gpu", "uuid": f"GPU-A{i}",
             "capacity": "49140Mi", "owner": [], "assigned": 0, "unassigned": 1, "nodeFlags": []} for i in range(2)]
    p = tmp_path / "pool.yaml"
    p.write_text(json.dumps({"kind": "GpuResourcePool", "metadata": {"name": "gpu-pool"},
                             "status": {"resources": res, "sourceVersions": {"pods": "1"}}}), encoding="utf-8")
    return p


def dra_request(pool, allocation, tp=2, gpu=1):
    return {"apiVersion": "grp.keti.re.kr/v1alpha1", "kind": "PlanRequest", "metadata": {"name": "w"},
            "spec": {"requestId": "req-dra-1", "workloadRef": "w", "runtime": "pytorch-3d",
                     "parallelism": {"dp": 1, "tp": tp, "pp": 1}, "poolSnapshot": str(pool),
                     **({"allocation": {"gpuIds": allocation}} if allocation else {}),
                     "launch": {"script": "workloads/train_3d.py", "gpu": gpu}}}


def test_dra_plan_pins_each_rank_to_allocated_device(tmp_path):
    from grp_runtime.execution_plan_controller import intake
    pool = dra_pool(tmp_path)
    res = intake.generate(dra_request(pool, ["GPU-A1", "GPU-A0"]))
    assert res.result == "Generated", res.message
    claims = res.decision["deviceBinding"]["claims"]
    assert res.decision["deviceBinding"]["mode"] == "dra"
    assert {c["uuid"] for c in claims} == {"GPU-A0", "GPU-A1"} and {c["node"] for c in claims} == {"gpu-node"}
    assert all(c["deviceClassName"] == "gpu.nvidia.com" for c in claims)
    assert res.manifest["spec"]["planning"]["deviceBinding"] == res.decision["deviceBinding"]
    mig = intake.generate(dra_request(pool, ["MIG-0", "MIG-2"])).decision["deviceBinding"]["claims"]
    assert [c["deviceClassName"] for c in mig] == ["mig.nvidia.com"] * 2
    assert {c["uuid"] for c in mig} == {"MIG-0", "MIG-2"}
    cpu = intake.generate(dra_request(pool, None, gpu=0)).decision["deviceBinding"]
    assert cpu["mode"] == "none" and "claims" not in cpu


def test_uneven_allocation_across_nodes_is_used_as_given(tmp_path):
    from grp_runtime.execution_plan_controller import intake
    doc = dra_request(dra_pool(tmp_path), ["MIG-0", "MIG-1", "MIG-2", "GPU-A0"])
    doc["spec"]["parallelism"] = {"dp": 1, "tp": 2, "pp": 2}
    res = intake.generate(doc)
    assert res.result == "Generated", res.message
    claims = res.decision["deviceBinding"]["claims"]
    assert {c["uuid"] for c in claims} == {"MIG-0", "MIG-1", "MIG-2", "GPU-A0"}
    assert {c["node"] for c in claims} == {"gpu-node-a", "gpu-node"}
    assert res.decision["communication"]["backend"] == "gloo"
    placement = res.manifest["spec"]["placement"]
    assert placement["perNode"] == {"gpu-node-a": 3, "gpu-node": 1}
    assert any("TP group(s) across nodes" in w for w in res.decision["warnings"])
    pool = GrpPoolStatusClient(str(tmp_path / "pool.yaml"))
    ep = {"metadata": {"namespace": "default"}, "spec": res.manifest["spec"]}
    out = validate(ep, pool)
    assert out["validation"] == "Accepted", out["reason"]
    assert out["placement"]["state"] == "Placed" and out["placement"]["rankSource"] == "plan"


def test_scheduler_allocation_waits_then_plans_from_reservation(tmp_path):
    from grp_runtime.execution_plan_controller import intake
    pool = dra_pool(tmp_path)
    doc = dra_request(pool, None)
    doc["spec"]["allocation"] = {"source": "scheduler"}
    wait = intake.generate(doc)
    assert wait.result == "AwaitingAllocation" and wait.manifest is None
    ask = wait.decision["allocationRequest"]
    assert ask["kind"] == "CandidateRequest" and ask["request_id"] == "req-dra-1"
    assert ask["workload"]["gpu_per_pod"] == 1 and ask["workload"]["replica"] == 2 and ask["rankShape"]["tp"] == 2
    assert intake.status_summary(wait)["allocationRequest"] == ask
    doc["spec"]["allocation"]["reservation"] = {"ok": True, "request_id": "req-dra-1", "candidate_id": "cand-1",
                                                "placements": [{"node_id": "gpu-node-a", "gpu_id": "GPU0",
                                                                "profile": "1g.24gb", "placement": p} for p in (3, 6)]}
    done = intake.generate(doc)
    assert done.result == "Generated", done.message
    assert done.decision["allocation"]["source"] == "reservation"
    assert {c["uuid"] for c in done.decision["deviceBinding"]["claims"]} == {"MIG-1", "MIG-2"}
    bad = dra_request(pool, None)
    bad["spec"].pop("parallelism")
    bad["spec"]["resources"] = {"gpus": 2}
    bad["spec"]["allocation"] = {"source": "scheduler"}
    assert intake.generate(bad).reason == "InvalidSpec"


def test_reservation_from_pool_controller_resolves_to_dra_devices(tmp_path):
    from grp_runtime.execution_plan_controller import intake
    pool = dra_pool(tmp_path)

    def req(slots, ok=True):
        doc = dra_request(pool, None)
        doc["spec"]["allocation"] = {"reservation": {
            "kind": "ReservationResult", "request_id": "sched-1", "candidate_id": "cand-abc", "ok": ok,
            "reason": None if ok else "GPU0 1g.24gb@0 is already in use",
            "placements": [{"node_id": n, "gpu_id": g, "profile": p, "placement": s} for n, g, p, s in slots]}}
        return doc

    res = intake.generate(req([("gpu-node", "GPU1", "full", 0), ("gpu-node", "GPU0", "full", 0)]))
    assert res.result == "Generated", res.message
    alloc = res.decision["allocation"]
    assert alloc["source"] == "reservation" and alloc["candidateId"] == "cand-abc"
    assert alloc["gpuIds"] == ["GPU-A1", "GPU-A0"]
    assert {c["uuid"] for c in res.decision["deviceBinding"]["claims"]} == {"GPU-A0", "GPU-A1"}
    assert not any("planner fallback" in w for w in res.decision["warnings"])
    mig = intake.generate(req([("gpu-node-a", "GPU0", "1g.24gb", 3), ("gpu-node-a", "GPU0", "1g.24gb", 9)]))
    assert {c["uuid"] for c in mig.decision["deviceBinding"]["claims"]} == {"MIG-1", "MIG-3"}
    gone = intake.generate(req([("gpu-node-a", "GPU0", "1g.24gb", 3), ("gpu-node-a", "GPU1", "1g.24gb", 0)]))
    assert gone.result == "Rejected" and gone.reason == "ReservationUnresolved" and "GPU1" in gone.message
    denied = intake.generate(req([("gpu-node", "GPU0", "full", 0), ("gpu-node", "GPU1", "full", 0)], ok=False))
    assert denied.result == "Rejected" and denied.reason == "InvalidSpec"
    short = intake.generate(req([("gpu-node", "GPU0", "full", 0)]))
    assert short.result == "Rejected" and short.reason == "InvalidSpec"
    doc = req([("gpu-node-a", "GPU7", "1g.24gb", 99), ("gpu-node", "GPU0", "full", 0)])
    doc["spec"]["allocation"]["reservation"]["placements"][0]["uuid"] = "MIG-2"
    by_uuid = intake.generate(doc)
    assert by_uuid.result == "Generated", by_uuid.message
    assert by_uuid.decision["allocation"]["gpuIds"] == ["MIG-2", "GPU-A0"]


def test_pool_controller_format_claim_owners_and_mig_start_from_device_name():
    from grp_runtime.pool import GrpPoolStatusClient
    res = [{"node": "w1", "resource": f"gpu-0-mig-1g24gb-14-{s}", "deviceType": "mig", "migProfile": "1g.24gb",
            "uuid": f"MIG-{s}", "parentUuid": "GPU-P", "capacity": "24576Mi", "path": "DRA",
            "owner": ["ai-training/job-g2-r1-gpu"] if s == 3 else [], "assigned": int(s == 3),
            "unassigned": int(s != 3), "nodeFlags": []} for s in (0, 3)]
    p = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool", {"metadata": {"name": "gpu-pool"},
                                                              "status": {"resources": res, "sourceVersions": {}}})
    assert [g.uuid for g in p.gpus_of("ai-training/job")] == ["MIG-3"]
    assert p.slot_uuids() == {("w1", "GPU0", "1g.24gb", 0): "MIG-0", ("w1", "GPU0", "1g.24gb", 3): "MIG-3"}


def test_gpu_plan_without_device_uuids_is_rejected():
    import copy
    import yaml
    from grp_runtime.execution_plan_controller import intake
    doc = yaml.safe_load((ROOT / "scenarios" / "plan-request" / "pr-train-fixed.yaml").read_text(encoding="utf-8"))
    doc = copy.deepcopy(doc)
    doc["spec"]["launch"]["gpu"] = 1
    res = intake.generate(doc)
    assert res.result == "Rejected" and res.reason == "DeviceUnresolved"
    auto = copy.deepcopy(doc)
    auto["spec"].pop("parallelism")
    auto["spec"]["resources"] = {"gpus": 2}
    assert intake.generate(auto).reason == "InvalidSpec"


class RecordingCustom:
    def __init__(self, items=()):
        self.created, self.items = [], list(items)

    def create_namespaced_custom_object(self, group, version, ns, plural, body):
        self.created.append((group, version, plural, body))

    def list_namespaced_custom_object(self, group, version, ns, plural, label_selector=None):
        return {"items": self.items}


def test_dra_launch_creates_one_claim_per_rank_and_binds_pods(tmp_path):
    from grp_runtime.execution_plan_controller import intake
    res = intake.generate(dra_request(dra_pool(tmp_path), ["GPU-A0", "GPU-A1"]))
    spec = res.manifest["spec"]
    core, custom = RecordingCore(), RecordingCustom()
    owner = {"apiVersion": "grp.keti.re.kr/v1alpha1", "kind": "ExecutionPlan", "name": "w", "uid": "u1"}
    nodes = [x["node"] for x in sorted(spec["planning"]["rankMap"], key=lambda x: x["rank"])]
    requested = Workload("default", "w", core=core, custom=custom).launch(
        2, nodes, "pytorch-3d", spec["launch"], 1, None, "t", parallelism=spec["parallelism"], owner=owner,
        plan=spec["planning"])
    by_rank = {c["rank"]: c for c in spec["planning"]["deviceBinding"]["claims"]}
    assert [(g, v, p) for g, v, p, _ in custom.created] == [("resource.k8s.io", "v1", "resourceclaims")] * 2
    for _, _, _, body in custom.created:
        r = int(body["metadata"]["labels"]["grp.keti.re.kr/rank"])
        req = body["spec"]["devices"]["requests"][0]["exactly"]
        assert req["count"] == 1 and req["deviceClassName"] == "gpu.nvidia.com"
        assert req["selectors"][0]["cel"]["expression"] == f'device.attributes["gpu.nvidia.com"].uuid == "{by_rank[r]["uuid"]}"'
        assert body["metadata"]["ownerReferences"] == [owner] and body["metadata"]["name"] == f"w-g1-r{r}-gpu"
    for pod in core.pods:
        r = int(pod.metadata.labels["grp.keti.re.kr/rank"])
        assert pod.spec.resource_claims[0].resource_claim_name == f"w-g1-r{r}-gpu"
        res_ = pod.spec.containers[0].resources
        assert "nvidia.com/gpu" not in res_.limits and res_.claims[0].name == "gpu"
    assert [x["plannedUuid"] for x in requested] == [by_rank[r]["uuid"] for r in range(2)]


def test_gpu_launch_refuses_plan_without_claims():
    import pytest
    with pytest.raises(ValueError, match="DRA device claim"):
        Workload("default", "w", core=RecordingCore(), custom=RecordingCustom()).launch(
            2, ["n", "n"], "ddp", {"script": "x.py", "gpu": 1}, 2, None, "t")


def claim(rank, uuid, device=None, pod=None):
    status = {}
    if device:
        status = {"allocation": {"devices": {"results": [{"request": "gpu", "driver": "gpu.nvidia.com",
                                                          "pool": "gpu-node", "device": device}]}},
                  "reservedFor": [{"resource": "pods", "name": pod}]}
    return {"metadata": {"name": f"w-g1-r{rank}-gpu", "labels": {"grp.keti.re.kr/rank": str(rank)},
                         "annotations": {"grp.keti.re.kr/device-uuid": uuid}}, "status": status}


def test_claim_binding_bound_only_when_every_rank_claim_is_allocated():
    from grp_runtime.runtime_coordination.runtime_action_controller.operator import _claim_binding
    spec = {"parallelism": {"dp": 1, "tp": 2, "pp": 1}, "launch": {"gpu": 1}}
    half = Workload("default", "w", core=RecordingCore(),
                    custom=RecordingCustom([claim(0, "GPU-A0", "gpu-0", "w-g1-r0"), claim(1, "GPU-A1")]))
    b = _claim_binding(spec, half, 1)
    assert b["state"] == "Pending" and "w-g1-r1-gpu" in b["reason"]
    full = Workload("default", "w", core=RecordingCore(), custom=RecordingCustom(
        [claim(1, "GPU-A1", "gpu-1", "w-g1-r1"), claim(0, "GPU-A0", "gpu-0", "w-g1-r0")]))
    b = _claim_binding(spec, full, 1)
    assert b["state"] == "Bound" and b["pool"] == "dra"
    assert [(d["rank"], d["resource"], d["uuid"], d["pod"]) for d in b["devices"]] == [
        (0, "gpu-0", "GPU-A0", "w-g1-r0"), (1, "gpu-1", "GPU-A1", "w-g1-r1")]


def test_observer_builds_pool_from_resourceslices_and_claims():
    import importlib.util
    from datetime import datetime, timezone
    from grp_runtime.pool import GrpPoolStatusClient
    spec_ = importlib.util.spec_from_file_location("gpu_pool_observer", ROOT / "deploy" / "pool" / "gpu_pool_observer.py")
    obs = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(obs)

    def dev(name, uuid, typ, profile=None, parent=None, mem="24192Mi"):
        attrs = {"uuid": {"string": uuid}, "type": {"string": typ}, "productName": {"string": "NVIDIA RTX PRO 6000"},
                 "resource.kubernetes.io/pciBusID": {"string": "0000:2a:00.0"}}
        if profile:
            attrs.update(profile={"string": profile}, parentUUID={"string": parent})
        cap = {"memory": {"value": mem}}
        if profile:
            cap[f"memorySlice{name.rsplit('-', 1)[1]}"] = {"value": "1"}
        return {"name": name, "attributes": attrs, "capacity": cap}

    slices = [{"metadata": {"name": "gpu-node-a-gpu"}, "spec": {
                  "driver": "gpu.nvidia.com", "nodeName": "gpu-node-a", "pool": {"name": "gpu-node-a"},
                  "devices": [dev(f"gpu-0-mig-1g24gb-14-{p}", f"MIG-{p}", "mig", "1g.24gb", "GPU-P") for p in (0, 3)]}},
              {"metadata": {"name": "other"}, "spec": {"driver": "other.example.com", "nodeName": "gpu-node-a",
                                                      "pool": {"name": "gpu-node-a"}, "devices": [dev("x", "X", "gpu")]}},
              {"metadata": {"name": "w2-gpu"}, "spec": {
                  "driver": "gpu.nvidia.com", "nodeName": "w2", "pool": {"name": "w2"},
                  "devices": [dev("gpu-0", "GPU-B", "gpu", mem="49140Mi")]}}]
    claims = [{"metadata": {"namespace": "default", "name": "t-g1-r0-gpu"},
               "status": {"allocation": {"devices": {"results": [
                   {"driver": "gpu.nvidia.com", "pool": "gpu-node-a", "device": "gpu-0-mig-1g24gb-14-3", "request": "gpu"}]}},
                   "reservedFor": [{"resource": "pods", "name": "t-g1-r0"}]}},
              {"metadata": {"namespace": "default", "name": "pending"}, "status": {}}]
    st = obs.build_status(slices, claims, {"gpu-node-a": [], "w2": []}, "gpu.nvidia.com", {"resourceClaims": "7"},
                          datetime.now(timezone.utc), 60)
    by = {r["resource"]: r for r in st["resources"]}
    assert set(by) == {"gpu-0-mig-1g24gb-14-0", "gpu-0-mig-1g24gb-14-3", "gpu-0"}
    assert by["gpu-0-mig-1g24gb-14-3"]["owner"] == ["default/t-g1-r0"] and by["gpu-0"]["unassigned"] == 1
    assert by["gpu-0-mig-1g24gb-14-0"]["deviceType"] == "mig" and by["gpu-0-mig-1g24gb-14-0"]["parentUuid"] == "GPU-P"
    assert st["summary"] == {"units": 3, "assigned": 1, "unassigned": 2, "unknown": 0}
    p = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool", {"metadata": {"name": "gpu-pool"}, "status": st})
    assert sorted(g.uuid for g in p.free_gpus()) == ["GPU-B", "MIG-0"]
    assert [g.uuid for g in p.gpus_of("default/t")] == ["MIG-3"]
    assert p.interconnect == {"gpu-node-a": "MIG", "w2": "unknown"}
    assert by["gpu-0-mig-1g24gb-14-3"]["placement"] == 3 and by["gpu-0"]["placement"] == 0
    assert p.slot_uuids() == {("gpu-node-a", "GPU0", "1g.24gb", 0): "MIG-0", ("gpu-node-a", "GPU0", "1g.24gb", 3): "MIG-3",
                              ("w2", "GPU0", "full", 0): "GPU-B"}


def test_launch_rejected_midway_is_cleaned_up(tmp_path):
    import pytest
    from kubernetes import client
    from grp_runtime.execution_plan_controller import intake
    from grp_runtime.runtime_adapter.workload import LaunchError
    res = intake.generate(dra_request(dra_pool(tmp_path), ["GPU-A0", "GPU-A1"]))
    spec = res.manifest["spec"]

    class QuotaCustom(RecordingCustom):
        def create_namespaced_custom_object(self, group, version, ns, plural, body):
            if self.created:
                e = client.ApiException(status=403, reason="Forbidden")
                e.body = json.dumps({"message": "exceeded quota: grp-runtime-quota"})
                raise e
            super().create_namespaced_custom_object(group, version, ns, plural, body)

    wl = Workload("default", "w", core=RecordingCore(), custom=QuotaCustom())
    torn = []
    wl.teardown = lambda timeout=60: torn.append(True)
    nodes = [x["node"] for x in sorted(spec["planning"]["rankMap"], key=lambda x: x["rank"])]
    with pytest.raises(LaunchError, match="exceeded quota"):
        wl.launch(2, nodes, "pytorch-3d", spec["launch"], 1, None, "t", parallelism=spec["parallelism"],
                  plan=spec["planning"])
    assert torn == [True]


def test_teardown_waits_for_kubelet_before_releasing_claims(monkeypatch):
    log = []

    class Core:
        def __init__(self):
            self.terminating = 2

        def list_namespaced_pod(self, ns, label_selector):
            log.append("list-pods")
            self.terminating -= 1
            return SimpleNamespace(items=[] if self.terminating < 0 else [SimpleNamespace(metadata=SimpleNamespace(name="w-g1-r0"))])

        def delete_namespaced_pod(self, name, ns, grace_period_seconds):
            log.append(("delete-pod", grace_period_seconds))

    class Custom:
        claims = 2

        def delete_collection_namespaced_custom_object(self, *a, label_selector):
            log.append("delete-claims")

        def list_namespaced_custom_object(self, *a, label_selector):
            log.append("list-claims")
            self.claims -= 1
            return {"items": [{}] * max(self.claims, 0)}

    monkeypatch.setattr(time, "sleep", lambda s: None)
    Workload("ns", "w", core=Core(), custom=Custom()).teardown()
    assert log[1] == ("delete-pod", 1)
    assert log.index("delete-claims") > max(i for i, x in enumerate(log) if x == "list-pods")
    assert log[-1] == "list-claims" and log.count("list-claims") == 2
