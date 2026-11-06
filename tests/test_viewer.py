import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from grp_runtime.viewer.__main__ import Viewer, handler
from grp_runtime.viewer.collect import summarize

MIG = ["MIG-a", "MIG-b", "MIG-c", "MIG-d"]


def _raw():
    ep = {"metadata": {"name": "p1", "creationTimestamp": "2026-11-06T01:00:00Z"},
          "spec": {"workloadRef": "w1", "runtime": "ddp", "parallelism": {"dp": 2, "tp": 1, "pp": 1},
                   "planning": {"requestRef": {"kind": "PlanRequest", "name": "p1"},
                                "rankMap": [{"rank": 1, "dp": 1, "tp": 0, "pp": 0, "node": "n1", "gpu": "n1:3",
                                             "uuid": MIG[3]},
                                            {"rank": 0, "dp": 0, "tp": 0, "pp": 0, "node": "n1", "gpu": "n1:2",
                                             "uuid": MIG[2]}],
                                "settings": {"source": "profile", "effectiveBatch": 1024,
                                             "batch": {"microBatch": 128, "gradientAccumulation": 4},
                                             "profiles": {"profileId": "dp2-m128", "throughput": 47038.0,
                                                          "status": "planned", "evaluated": 11, "matched": 4,
                                                          "feasible": 4, "origin": "configmap/x@1"}}}},
          "status": {"state": "Ready", "generation": 2, "assignedNodes": ["n1"],
                     "history": [{"event": "launch", "generation": 1}, {"event": "reconfigure", "generation": 2,
                                                                         "action": "ra1"}]}}
    ra = lambda name, created, action: {
        "metadata": {"name": name, "creationTimestamp": created},
        "spec": {"executionPlanRef": "p1", "action": action, "target": {"gpus": 2}},
        "status": {"phase": "Succeeded", "previous": None, "before": {"dp": 4, "gpuCount": 4, "step": 318},
                   "after": {"dp": 2, "gpuCount": 2, "resumedStep": 319},
                   "conditions": [{"phase": "Accepted", "time": created}]}}
    pool = {"metadata": {"name": "gpu-pool"},
            "status": {"observedAt": "2026-11-06T01:10:00Z", "resources": [
                {"node": "n1", "resource": f"gpu-0-mig-{i}", "uuid": u, "parentUuid": "GPU-0", "deviceType": "mig",
                 "migProfile": "1g.24gb", "model": "RTX PRO 6000",
                 **({"owner": ["grp-runtime/w1-g2-r0"]} if i == 2 else {"unassigned": True})}
                for i, u in enumerate(MIG)]}}
    return {"executionplans": [ep],
            "runtimeactions": [ra("ra2", "2026-11-06T01:20:00Z", "Observe"),
                               ra("ra1", "2026-11-06T01:05:00Z", "ScaleDown"),
                               ra("other", "2026-11-06T01:06:00Z", "Observe") | {"spec": {"executionPlanRef": "p9"}}],
            "planrequests": [{"metadata": {"name": "p1"}},
                             {"metadata": {"name": "p2"}, "status": {"phase": "AwaitingAllocation"},
                              "spec": {"parallelism": {"dp": 1, "tp": 1, "pp": 1}}}],
            "progress": {"w1": {"status": "running", "step": 400, "stepsPerSec": 45.9, "resumedStep": 319,
                                "appliedSettings": {"effectiveBatch": 1024, "microBatch": 128,
                                                    "gradientAccumulation": 4}, "rankPids": [1, 2]}},
            "pool": pool}


def test_summarize_links_plan_actions_progress_and_pool():
    out = summarize(_raw(), {"p1": [[1.0, 399, 45.0, 2, "running"]]})
    (w,) = out["workloads"]
    assert [r["rank"] for r in w["ranks"]] == [0, 1]
    assert [a["name"] for a in w["actions"]] == ["ra1", "ra2"]
    assert w["actions"][0]["after"]["resumedStep"] == 319
    assert w["settings"]["profileId"] == "dp2-m128" and w["settings"]["matched"] == 4
    assert w["progress"]["step"] == 400 and "rankPids" not in w["progress"]
    assert w["samples"] == [[1.0, 399, 45.0, 2, "running"]]
    assert [r["name"] for r in out["requests"]] == ["p2"]
    devices = {d["uuid"]: d for d in out["pool"]["devices"]}
    assert devices[MIG[2]]["workload"] == "p1" and devices[MIG[3]]["workload"] == "p1"
    assert devices[MIG[0]]["workload"] is None and devices[MIG[0]]["holder"] is None
    assert devices[MIG[2]]["holder"] == "grp-runtime/w1"


def test_summarize_without_pool_or_settings():
    raw = _raw()
    raw["pool"] = None
    del raw["executionplans"][0]["spec"]["planning"]["settings"]
    w = summarize(raw)["workloads"][0]
    assert w["settings"] is None and w["samples"] == []
    assert summarize(raw)["pool"] is None


def test_viewer_serves_page_and_state(tmp_path):
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps(_raw()), encoding="utf-8")
    v = Viewer("grp-runtime", "gpu-pool", 2.0, 10, str(snap))
    v.tick()
    v.tick()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler(v))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        state = json.loads(urllib.request.urlopen(f"{base}/api/state").read())
        page = urllib.request.urlopen(f"{base}/").read().decode()
    finally:
        srv.shutdown()
    assert state["error"] is None and state["workloads"][0]["name"] == "p1"
    assert len(state["workloads"][0]["samples"]) == 2
    assert "GRP 런타임 엔진" in page and "api/state" in page


def test_viewer_reports_fetch_errors(tmp_path):
    v = Viewer("grp-runtime", "gpu-pool", 2.0, 10)
    v.raw = lambda: (_ for _ in ()).throw(RuntimeError("forbidden"))
    v.tick()
    body = json.loads(v.body)
    assert body["error"] == "forbidden" and body["workloads"] == []
