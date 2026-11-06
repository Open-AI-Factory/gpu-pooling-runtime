import importlib.util
from pathlib import Path

from grp_runtime.execution_plan_controller.profiles import select, validate_profiles
from grp_runtime.pool import GrpPoolStatusClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measure_profiles", ROOT / "deploy/profiles/measure_profiles.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)

MODEL = "RTX PRO 6000 Blackwell Server Edition"
RES = [{"node": "gpu-node-a", "resource": f"gpu-0-mig-1g24gb-14-{i}", "deviceType": "mig", "migProfile": "1g.24gb",
        "model": MODEL, "uuid": f"MIG-{i}", "parentUuid": "GPU-A", "capacity": "24192Mi", "unassigned": 1,
        "nodeFlags": []} for i in range(4)]
IDENT = {"signature": "workloads/train_ddp.py", "model": "grp/train_ddp-mlp", "precision": "float32",
         "runtime": {"name": "pytorch", "version": "2.11"}}


def measured(throughput, peak, first=0.5, last=0.3):
    return {"throughput": throughput, "measuredSteps": 50, "peakMemoryBytesPerDevice": peak,
            "firstLoss": first, "lastLoss": last}


def test_measured_bundle_feeds_the_selector():
    hw, per_node = mp.hardware(RES, ["MIG-0", "MIG-1"])
    assert hw["deviceMemoryBytes"] == 24192 * 2 ** 20 and per_node == {"gpu-node-a": 2}
    good = mp.build_profile(measured(9000.0, 300 * 2 ** 20), 2, 128, 4, 1024, hw, per_node, IDENT, "k8s:a", 1.2)
    fast = mp.build_profile(measured(12000.0, 500 * 2 ** 20), 2, 512, 1, 1024, hw, per_node, IDENT, "k8s:b", 1.2)
    bad = mp.build_profile(measured(15000.0, 500 * 2 ** 20, 0.3, 0.9), 2, 256, 2, 1024, hw, per_node, IDENT, "k8s:c", 1.2)
    assert good["config"]["memoryBudgetBytesPerDevice"] == 360 * 2 ** 20
    assert not bad["metrics"]["qualityPassed"]
    b = mp.bundle([good, fast, bad], [], IDENT["signature"], 1000.0, 30)
    assert validate_profiles(b["acceptedProfiles"]) and len(b["excludedProfiles"]) == 1
    pool = GrpPoolStatusClient("k8s:gpuresourcepools/gpu-pool",
                               {"metadata": {"name": "gpu-pool"}, "status": {"resources": RES, "sourceVersions": {}}})
    s, why = select({"settings": {"effectiveBatch": 1024, "maxMicroBatch": 512, "profiles": b}},
                    {"dp": 2, "tp": 1, "pp": 1}, list(pool.gpus.values())[:2])
    assert not why and s["profiles"]["profileId"] == fast["id"]


def test_mixed_hardware_is_refused():
    other = dict(RES[1], model="RTX A6000", deviceType="gpu", uuid="GPU-B")
    try:
        mp.hardware([RES[0], other], ["MIG-0", "GPU-B"])
    except ValueError as e:
        assert "differ" in str(e)
    else:
        raise AssertionError


def test_measurement_request_pins_the_batch():
    tpl = {"apiVersion": "grp.keti.re.kr/v1alpha1", "kind": "PlanRequest", "metadata": {"name": "x"},
           "spec": {"workloadRef": "x", "launch": {"script": "workloads/train_ddp.py", "args": ["--step-sleep", "1"]}}}
    pr = mp.render_request(tpl, "prof-d2-m128", 2, 128, 4, 1024, ["MIG-0", "MIG-1", "MIG-2"], 60, 10)
    assert pr["spec"]["allocation"]["gpuIds"] == ["MIG-0", "MIG-1"]
    assert pr["spec"]["launch"]["args"] == ["--max-steps", "60", "--measure-warmup", "10"]
    s, _ = select(pr["spec"]["launch"], {"dp": 2, "tp": 1, "pp": 1})
    assert s["source"] == "pinned" and s["batch"] == {"microBatch": 128, "gradientAccumulation": 4}
    assert select(pr["spec"]["launch"], {"dp": 4, "tp": 1, "pp": 1})[1]
    assert mp.parse_configs("4x64, 2X512") == [(4, 64), (2, 512)]
