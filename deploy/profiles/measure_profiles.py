import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import yaml

GROUP, VERSION = "grp.keti.re.kr", "v1alpha1"
POOL = ("gpu.platform.io", "v1alpha1", "gpuresourcepools")
PROVIDER = "grp-runtime-measure"
UNITS = {"throughput": "samples/s", "peakMemoryBytesPerDevice": "bytes", "timestamps": "unix-seconds"}


def parse_configs(text: str) -> list[tuple[int, int]]:
    out = []
    for item in text.split(","):
        dp, micro = item.strip().lower().split("x")
        out.append((int(dp), int(micro)))
    return out


def mi_bytes(capacity: str | None) -> int:
    units = {"Ki": 2 ** 10, "Mi": 2 ** 20, "Gi": 2 ** 30, "Ti": 2 ** 40}
    for suf, mul in units.items():
        if capacity and capacity.endswith(suf):
            return int(float(capacity[:-len(suf)]) * mul)
    return int(capacity or 0)


def hardware(resources: list[dict], uuids: list[str]) -> tuple[dict, dict[str, int]]:
    by_uuid = {r.get("uuid"): r for r in resources}
    missing = [u for u in uuids if u not in by_uuid]
    if missing:
        raise ValueError(f"devices not in pool: {missing}")
    hws, per_node = set(), {}
    for u in uuids:
        r = by_uuid[u]
        mig = r.get("deviceType") == "mig"
        hws.add((r.get("model") or "", "mig" if mig else "whole", (r.get("migProfile") or "") if mig else "",
                 mi_bytes(r.get("capacity"))))
        per_node[r.get("node", "")] = per_node.get(r.get("node", ""), 0) + 1
    if len(hws) != 1:
        raise ValueError(f"devices differ in hardware: {sorted(hws)}")
    model, mode, mig_profile, mem = hws.pop()
    return ({"model": model, "mode": mode, "migProfile": mig_profile, "deviceMemoryBytes": mem,
             "sharing": "exclusive"}, per_node)


def build_profile(measurement: dict, dp: int, micro: int, accum: int, total: int, hw: dict,
                  per_node: dict[str, int], identity: dict, ref: str, headroom: float) -> dict:
    counts = set(per_node.values())
    if len(counts) != 1:
        raise ValueError(f"uneven placement {per_node}")
    peak = int(measurement["peakMemoryBytesPerDevice"])
    budget = min(hw["deviceMemoryBytes"] or 2 ** 62, max(peak, math.ceil(peak * headroom / 2 ** 20) * 2 ** 20))
    first, last = measurement.get("firstLoss"), measurement.get("lastLoss")
    quality = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (first, last)) and last < first
    return {"id": f"{identity['signature'].rsplit('/', 1)[-1].removesuffix('.py')}-dp{dp}-m{micro}x{accum}-"
                  f"{(hw['migProfile'] or hw['model']).replace(' ', '_').lower()}",
            "kind": "training", "runtime": dict(identity["runtime"]), "model": {"id": identity["model"]},
            "precision": identity["precision"],
            "workload": {"signature": identity["signature"], "effectiveBatch": total},
            "hardware": hw,
            "config": {"batch": {"microBatch": micro, "gradientAccumulation": accum},
                       "parallelism": {"dp": dp, "tp": 1, "pp": 1},
                       "placement": {"nodeCount": len(per_node), "gpusPerNode": counts.pop(), "interconnect": "none"},
                       "memoryBudgetBytesPerDevice": budget},
            "metrics": {"throughput": round(float(measurement["throughput"]), 3), "latencyP95Ms": 0.0,
                        "peakMemoryBytesPerDevice": peak, "qualityPassed": quality},
            "sampleCount": int(measurement["measuredSteps"]),
            "evidence": {"kind": "measured", "ref": ref}}


def bundle(profiles: list[dict], failed: list[dict], signature: str, now: float, ttl_days: int) -> dict:
    return {"schemaVersion": 1, "providerId": PROVIDER, "bundleId": signature,
            "revision": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now)),
            "createdAt": int(now), "expiresAt": int(now + ttl_days * 86400), "units": UNITS,
            "profiles": profiles,
            "acceptedProfiles": [p for p in profiles if p["metrics"]["qualityPassed"]],
            "excludedProfiles": [{"profile": p, "reason": "quality_failed"}
                                 for p in profiles if not p["metrics"]["qualityPassed"]],
            "evidenceManifest": [{"profileId": p["id"], "artifactRef": p["evidence"]["ref"], "measuredAt": int(now),
                                  "qualityVerdict": "passed" if p["metrics"]["qualityPassed"] else "failed"}
                                 for p in profiles],
            "failedRuns": failed}


def render_request(template: dict, name: str, dp: int, micro: int, accum: int, total: int, uuids: list[str],
                   steps: int, warmup: int) -> dict:
    pr = copy.deepcopy(template)
    pr["metadata"] = {"name": f"{name}-plan"}
    spec = pr["spec"]
    spec.update(requestId=f"req-{name}", workloadRef=name, parallelism={"dp": dp, "tp": 1, "pp": 1})
    spec["allocation"] = {"gpuIds": uuids[:dp]}
    launch = spec.setdefault("launch", {})
    launch["settings"] = {"effectiveBatch": total, "maxMicroBatch": micro,
                          "batch": {"microBatch": micro, "gradientAccumulation": accum}}
    launch["args"] = ["--max-steps", str(steps), "--measure-warmup", str(warmup)]
    return pr


def wait_for(fn, timeout: float, every: float = 3.0):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(every)
    return None


def pool_holds(co, pool: str, ns: str, wl: str) -> bool:
    res = co.get_cluster_custom_object(*POOL[:2], POOL[2], pool).get("status", {}).get("resources") or []
    return any(str(o).startswith(f"{ns}/{wl}-") for r in res for o in (r.get("owner") or []))


def run_one(co, core, ns: str, pr: dict, timeout: float, pool: str) -> tuple[dict | None, str]:
    name, wl = pr["metadata"]["name"], pr["spec"]["workloadRef"]
    co.create_namespaced_custom_object(GROUP, VERSION, ns, "planrequests", pr)

    def done():
        st = (co.get_namespaced_custom_object(GROUP, VERSION, ns, "planrequests", name).get("status") or {})
        if st.get("result") in ("Rejected", "Failed"):
            return {"error": f"{st.get('result')} {st.get('reason')}: {st.get('message')}"}
        try:
            raw = (core.read_namespaced_config_map(f"grp-{wl}", ns).data or {}).get("state")
        except Exception:
            return None
        state = json.loads(raw) if raw else {}
        if state.get("status") == "finished":
            return state if state.get("measurement") else {"error": "finished without measurement"}
        return None

    try:
        state = wait_for(done, timeout)
    finally:
        co.delete_namespaced_custom_object(GROUP, VERSION, ns, "planrequests", name)
        wait_for(lambda: not [p for p in core.list_namespaced_pod(ns).items if p.metadata.name.startswith(f"{wl}-")],
                 180)
        from kubernetes import client
        rc = client.ResourceV1Api() if hasattr(client, "ResourceV1Api") else None
        if rc is not None:
            wait_for(lambda: not [c for c in rc.list_namespaced_resource_claim(ns).items
                                  if c.metadata.name.startswith(f"{wl}-")], 180)
        wait_for(lambda: not pool_holds(co, pool, ns, wl), 180)
    if state is None:
        return None, f"timeout after {timeout}s"
    if "error" in state:
        return None, state["error"]
    return state, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--effective-batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--runtime-version", default="2.11")
    ap.add_argument("--pool", default="gpu-pool")
    ap.add_argument("--headroom", type=float, default=1.2)
    ap.add_argument("--ttl-days", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--prefix", default="prof")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configmap", default=None)
    a = ap.parse_args()

    from kubernetes import client, config
    config.load_kube_config()
    co, core = client.CustomObjectsApi(), client.CoreV1Api()
    template = yaml.safe_load(Path(a.template).read_text(encoding="utf-8"))
    uuids = [u.strip() for u in a.gpus.split(",") if u.strip()]
    signature = template["spec"]["launch"]["script"]
    identity = {"signature": signature, "model": a.model_id, "precision": a.precision,
                "runtime": {"name": "pytorch", "version": a.runtime_version}}
    resources = co.get_cluster_custom_object(*POOL[:2], POOL[2], a.pool).get("status", {}).get("resources") or []
    profiles, failed = [], []
    for dp, micro in parse_configs(a.configs):
        per_rank = a.effective_batch // dp
        if a.effective_batch % dp or per_rank % micro or dp > len(uuids):
            failed.append({"dp": dp, "microBatch": micro, "reason": "configuration does not divide the batch or GPUs"})
            continue
        accum = per_rank // micro
        hw, per_node = hardware(resources, uuids[:dp])
        name = f"{a.prefix}-d{dp}-m{micro}"
        print(f"[measure] {name}: DP{dp} micro {micro} x accum {accum}", flush=True)
        pr = render_request(template, name, dp, micro, accum, a.effective_batch, uuids, a.steps, a.warmup)
        state, why = run_one(co, core, a.namespace, pr, a.timeout, a.pool)
        if state is None:
            print(f"[measure] {name}: failed ({why})", flush=True)
            failed.append({"dp": dp, "microBatch": micro, "reason": why})
            continue
        if state.get("appliedSettings", {}).get("microBatch") != micro:
            failed.append({"dp": dp, "microBatch": micro, "reason": f"applied {state.get('appliedSettings')}"})
            continue
        m = state["measurement"]
        ref = f"k8s:{a.namespace}/{name}@{int(time.time())}"
        p = build_profile(m, dp, micro, accum, a.effective_batch, hw, per_node, identity, ref, a.headroom)
        print(f"[measure] {name}: {p['metrics']['throughput']} samples/s, peak "
              f"{p['metrics']['peakMemoryBytesPerDevice'] / 2 ** 20:.0f}MiB, quality {p['metrics']['qualityPassed']}",
              flush=True)
        profiles.append(p)
    b = bundle(profiles, failed, signature, time.time(), a.ttl_days)
    Path(a.out).write_text(json.dumps(b, indent=1), encoding="utf-8")
    if a.configmap:
        body = {"apiVersion": "v1", "kind": "ConfigMap",
                "metadata": {"name": a.configmap, "labels": {"grp.keti.re.kr/profiles": PROVIDER}},
                "data": {"profiles.json": json.dumps(b)}}
        try:
            core.replace_namespaced_config_map(a.configmap, a.namespace, body)
        except client.ApiException as e:
            if e.status != 404:
                raise
            core.create_namespaced_config_map(a.namespace, body)
    print(f"[measure] {len(b['acceptedProfiles'])} accepted, {len(b['excludedProfiles'])} excluded, "
          f"{len(failed)} failed -> {a.out}", flush=True)
    return 0 if profiles else 1


if __name__ == "__main__":
    sys.exit(main())
