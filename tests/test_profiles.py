import json
from copy import deepcopy

import pytest

from grp_runtime.execution_plan_controller.profiles import ProfileError, select, validate_profiles
from grp_runtime.pool import Gpu, GrpPoolStatusClient

MIG = "RTX PRO 6000 Blackwell Server Edition"


def mig(i, node="gpu-node-a"):
    return Gpu(f"{node}:{i}", node, 24.0, None, f"MIG-{i}", f"MIG-{i}", model=MIG, mig_profile="1g.24gb")


def profile(pid, dp, micro, accum, throughput, peak=10 * 2 ** 30, budget=20 * 2 ** 30, **over):
    p = {"id": pid, "kind": "training", "runtime": {"name": "pytorch", "version": "2.11"},
         "model": {"id": "toy/mlp"}, "precision": "float32",
         "workload": {"signature": "train_ddp", "effectiveBatch": 1024},
         "hardware": {"model": "NVIDIA " + MIG, "mode": "mig", "migProfile": "1g.24gb",
                      "deviceMemoryBytes": 24 * 2 ** 30, "sharing": "exclusive"},
         "config": {"batch": {"microBatch": micro, "gradientAccumulation": accum},
                    "parallelism": {"dp": dp, "tp": 1, "pp": 1},
                    "placement": {"nodeCount": 1, "gpusPerNode": dp, "interconnect": "none"},
                    "memoryBudgetBytesPerDevice": budget},
         "metrics": {"throughput": throughput, "latencyP95Ms": 1.0, "peakMemoryBytesPerDevice": peak,
                     "qualityPassed": True},
         "sampleCount": 3, "evidence": {"kind": "measured", "ref": f"mlflow:run-{pid}"}}
    for k, v in over.items():
        p[k] = v
    return p


PROFILES = [profile("dp4-256", 4, 256, 1, 900.0), profile("dp4-128", 4, 128, 2, 950.0),
            profile("dp2-256", 2, 256, 2, 600.0), profile("dp2-512", 2, 512, 1, 700.0)]


def launch(**extra):
    return {"settings": {"effectiveBatch": 1024, "maxMicroBatch": 512, "profiles": deepcopy(PROFILES), **extra}}


def test_measured_profile_wins_over_rule_for_the_allocated_hardware():
    s, why = select(launch(), {"dp": 4, "tp": 1, "pp": 1}, [mig(i) for i in range(4)])
    assert not why
    assert s["source"] == "profile" and s["profiles"]["profileId"] == "dp4-128"
    assert s["batch"] == {"microBatch": 128, "gradientAccumulation": 2}
    assert s["memoryBudgetBytesPerDevice"] == 20 * 2 ** 30
    assert s["profiles"]["matched"] == 2 and s["profiles"]["evidence"]["ref"] == "mlflow:run-dp4-128"


def test_reconfigure_reselects_profile_for_new_parallelism():
    s, _ = select(launch(), {"dp": 2, "tp": 1, "pp": 1}, [mig(1), mig(3)])
    assert s["profiles"]["profileId"] == "dp2-512"
    assert s["batch"]["microBatch"] * s["batch"]["gradientAccumulation"] * 2 == 1024


def test_micro_batch_cap_excludes_profiles():
    s, _ = select(launch(maxMicroBatch=256), {"dp": 2, "tp": 1, "pp": 1}, [mig(1), mig(3)])
    assert s["profiles"]["profileId"] == "dp2-256"


def test_unmeasured_hardware_falls_back_to_rule_and_says_so():
    a6000 = Gpu("w903:0", "w903", 48.0, None, "GPU-X", "GPU-X", model="RTX A6000")
    s, why = select(launch(), {"dp": 1, "tp": 1, "pp": 1}, [a6000])
    assert not why and s["source"] == "rule"
    assert s["batch"] == {"microBatch": 512, "gradientAccumulation": 2}
    assert s["profiles"]["status"] == "needsProfiling" and "RTX A6000" in s["profiles"]["reason"]


def test_matched_but_infeasible_profiles_reject_the_plan():
    bad = [profile("oom", 2, 512, 1, 700.0, budget=30 * 2 ** 30, peak=29 * 2 ** 30),
           profile("q", 2, 256, 2, 600.0, metrics={"throughput": 600.0, "latencyP95Ms": 1.0,
                                                   "peakMemoryBytesPerDevice": 1, "qualityPassed": False})]
    s, why = select({"settings": {"effectiveBatch": 1024, "maxMicroBatch": 512, "profiles": bad}},
                    {"dp": 2, "tp": 1, "pp": 1}, [mig(0), mig(1)])
    assert s is None and why.startswith("noFeasiblePlan")
    assert "budget_over_device_memory" in why and "quality_failed" in why


def test_identity_and_synthetic_evidence_are_respected():
    s, _ = select(launch(identity={"precision": "bfloat16"}), {"dp": 4, "tp": 1, "pp": 1}, [mig(i) for i in range(4)])
    assert s["source"] == "rule" and s["profiles"]["status"] == "needsProfiling"
    syn = [profile("syn", 4, 256, 1, 999.0, evidence={"kind": "synthetic", "ref": "test"})]
    base = {"effectiveBatch": 1024, "maxMicroBatch": 512, "profiles": syn}
    assert select({"settings": base}, {"dp": 4, "tp": 1, "pp": 1}, [mig(i) for i in range(4)])[0] is None
    s, _ = select({"settings": {**base, "allowSynthetic": True}}, {"dp": 4, "tp": 1, "pp": 1}, [mig(i) for i in range(4)])
    assert s["profiles"]["profileId"] == "syn"


def test_ep3_bundle_from_configmap_is_accepted():
    bundle = {"providerId": "ep3", "acceptedProfiles": deepcopy(PROFILES), "excludedProfiles": []}
    reads = []

    def reader(name, ns):
        reads.append((name, ns))
        return {"data": {"profiles.json": json.dumps(bundle)}, "resourceVersion": "42"}

    s, why = select({"settings": {"effectiveBatch": 1024, "maxMicroBatch": 512, "profileConfigMap": "prof"}},
                    {"dp": 4, "tp": 1, "pp": 1}, [mig(i) for i in range(4)], "team-a", reader)
    assert not why and reads == [("prof", "team-a")]
    assert s["profiles"]["origin"] == "configmap/prof@42" and s["profiles"]["profileId"] == "dp4-128"
    _, why = select({"settings": {"effectiveBatch": 1024, "maxMicroBatch": 512, "profileConfigMap": "prof"}},
                    {"dp": 4, "tp": 1, "pp": 1}, [], "team-a", lambda n, ns: {"data": {}})
    assert "has no profiles.json" in why


def test_malformed_profiles_are_rejected():
    broken = deepcopy(PROFILES[0])
    del broken["config"]["batch"]["gradientAccumulation"]
    with pytest.raises(ProfileError):
        validate_profiles([broken])
    dup = [PROFILES[0], PROFILES[0]]
    _, why = select({"settings": {"effectiveBatch": 1024, "maxMicroBatch": 512, "profiles": dup}},
                    {"dp": 4, "tp": 1, "pp": 1}, [mig(i) for i in range(4)])
    assert "unique" in why


def test_live_pool_carries_model_and_mig_profile():
    doc = {"kind": "GpuResourcePool", "metadata": {"name": "gpu-pool"}, "status": {"sourceVersions": {}, "resources": [
        {"node": "n", "resource": "gpu-0-mig-1g24gb-0", "deviceType": "mig", "migProfile": "1g.24gb", "model": MIG,
         "uuid": "MIG-0", "parentUuid": "GPU-A", "capacity": "24576Mi", "unassigned": 1, "nodeFlags": []},
        {"node": "n", "resource": "gpu-1", "deviceType": "gpu", "migProfile": "1g.24gb", "model": "RTX A6000",
         "uuid": "GPU-B", "capacity": "49140Mi", "unassigned": 1, "nodeFlags": []}]}}
    p = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool", doc)
    assert [(g.model, g.mig_profile) for g in p.gpus.values()] == [(MIG, "1g.24gb"), ("RTX A6000", "")]


def test_replan_on_reconfigure_uses_profiles_for_the_kept_devices():
    from grp_runtime.models import ParallelConfig, RuntimeActionSpec
    from test_k8s_logic import k8s_adapter, pool_doc
    doc = pool_doc([f"default/w-g1-r{i}" for i in range(4)])
    for r in doc["status"]["resources"]:
        r["model"] = MIG
    pool = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool", doc)
    ep = {"metadata": {"name": "w", "uid": "u"}, "spec": {"runtime": "pytorch", "launch": {"gpu": 1, **launch()},
          "parallelism": {"dp": 4, "tp": 1, "pp": 1}, "planning": {"selected": {"dp": 4, "tp": 1, "pp": 1}}},
          "status": {"generation": 1}}
    a = k8s_adapter(pool, ep)
    spec = RuntimeActionSpec.from_manifest({"metadata": {"name": "ra"}, "spec": {"workloadRef": "w", "target": {"gpus": 2}}})
    keep = [g.id for g in pool.gpus_of("default/w")][1::2]
    a._relaunch(spec, ParallelConfig(2, 1, 1), keep, "/ckpt", "ra", "reconfigure 4→2 ranks")
    settings = a.patches[-1][0]["planning"]["settings"]
    assert settings["source"] == "profile" and settings["profiles"]["profileId"] == "dp2-512"
    assert a.wl.calls[-1][5]["settings"]["batch"] == {"microBatch": 512, "gradientAccumulation": 1}


def test_model_names_match_across_pool_spellings():
    a6000 = Gpu("g:0", "g", 48.0, None, "GPU-X", "GPU-X", model="RTX-A6000")
    p = profile("a6000", 1, 512, 2, 1.0)
    p["hardware"] = {"model": "NVIDIA RTX A6000", "mode": "whole", "migProfile": "", "deviceMemoryBytes": 48 * 2 ** 30}
    p["config"]["memoryBudgetBytesPerDevice"] = 20 * 2 ** 30
    s, _ = select({"settings": {"effectiveBatch": 1024, "maxMicroBatch": 512, "profiles": [p]}},
                  {"dp": 1, "tp": 1, "pp": 1}, [a6000])
    assert s["source"] == "profile"
