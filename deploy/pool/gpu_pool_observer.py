import argparse
import time
from datetime import datetime, timedelta, timezone

from kubernetes import client, config

GROUP, VERSION, PLURAL = "gpu.platform.io", "v1alpha1", "gpuresourcepools"
DRA = ("resource.k8s.io", "v1")


def iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def _attrs(device: dict, driver: str) -> dict:
    out = {}
    for k, v in (device.get("attributes") or {}).items():
        out[k[len(driver) + 1:] if k.startswith(driver + "/") else k] = next(iter(v.values()), None) if v else None
    return out


def _placement(device: dict, mig: bool) -> int | None:
    if not mig:
        return 0
    slices = [int(k[len("memorySlice"):]) for k in (device.get("capacity") or {}) if k.startswith("memorySlice")]
    return min(slices) if slices else None


def held_devices(claims: list[dict]) -> dict[tuple, list[str]]:
    held = {}
    for c in claims:
        md, st = c["metadata"], c.get("status") or {}
        pods = [f"{md['namespace']}/{r['name']}" for r in st.get("reservedFor") or [] if r.get("resource") == "pods"]
        for r in ((st.get("allocation") or {}).get("devices") or {}).get("results") or []:
            held[(r["driver"], r["pool"], r["device"])] = pods or [f"{md['namespace']}/claim:{md['name']}"]
    return held


def node_flags(nodes: list) -> dict[str, list[str]]:
    flags = {}
    for n in nodes:
        f = []
        if not any(c.type == "Ready" and c.status == "True" for c in n.status.conditions or []):
            f.append("NotReady")
        if any(t.effect == "NoSchedule" for t in n.spec.taints or []):
            f.append("tainted")
        flags[n.metadata.name] = f
    return flags


def build_status(slices: list[dict], claims: list[dict], flags: dict[str, list[str]], driver: str,
                 versions: dict[str, str], started: datetime, ttl: int) -> dict:
    held = held_devices(claims)
    resources = []
    for s in sorted(slices, key=lambda s: (s["spec"].get("nodeName") or "", s["metadata"]["name"])):
        spec = s["spec"]
        if spec.get("driver") != driver:
            continue
        pool = (spec.get("pool") or {}).get("name") or spec.get("nodeName")
        node = spec.get("nodeName") or pool
        for d in spec.get("devices") or []:
            a = _attrs(d, driver)
            mem = ((d.get("capacity") or {}).get("memory") or {}).get("value")
            owner = held.get((driver, pool, d["name"]), [])
            resources.append({"resource": d["name"], "deviceType": a.get("type", "gpu"), "migProfile": a.get("profile"),
                              "placement": _placement(d, a.get("type") == "mig"),
                              "model": (a.get("productName") or "").replace("NVIDIA ", "") or None,
                              "uuid": a.get("uuid"), "parentUuid": a.get("parentUUID"), "capacity": mem,
                              "node": node, "path": "dra", "dra": {"driver": driver, "pool": pool, "device": d["name"]},
                              "units": 1, "assigned": 1 if owner else 0, "unassigned": 0 if owner else 1,
                              "owner": owner, "nodeFlags": flags.get(node, ["NotFound"]), "note": ""})
    assigned = sum(r["assigned"] for r in resources)
    observed = datetime.now(timezone.utc)
    return {
        "phase": "Observed", "atomicSnapshot": False,
        "collectionStartedAt": iso(started), "observedAt": iso(observed), "lastAttemptAt": iso(observed),
        "validUntil": iso(started + timedelta(seconds=ttl)),
        "nodeCount": len(flags), "gpuNodeCount": len({r["node"] for r in resources}),
        "resources": resources,
        "summary": {"units": len(resources), "assigned": assigned, "unassigned": len(resources) - assigned,
                    "unknown": 0},
        "sourceVersions": versions,
        "warnings": [] if resources else [f"no ResourceSlice devices for driver {driver}"],
    }


def observe(core: client.CoreV1Api, co: client.CustomObjectsApi, driver: str, ttl: int) -> dict:
    started = datetime.now(timezone.utc)
    slices = co.list_cluster_custom_object(*DRA, "resourceslices")
    claims = co.list_cluster_custom_object(*DRA, "resourceclaims")
    nodes = core.list_node()
    versions = {"resourceSlices": slices["metadata"]["resourceVersion"],
                "resourceClaims": claims["metadata"]["resourceVersion"], "nodes": nodes.metadata.resource_version}
    return build_status(slices["items"], claims["items"], node_flags(nodes.items), driver, versions, started, ttl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="gpu-pool")
    ap.add_argument("--driver", default="gpu.nvidia.com")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--ttl", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    core, co = client.CoreV1Api(), client.CustomObjectsApi()
    last = None
    while True:
        try:
            st = observe(core, co, a.driver, a.ttl)
            co.patch_cluster_custom_object_status(GROUP, VERSION, PLURAL, a.pool, {"status": st})
            key = [(r["node"], r["resource"], tuple(r["owner"])) for r in st["resources"]]
            if key != last:
                print(f"[{st['observedAt']}] {a.pool}: units={st['summary']['units']} "
                      f"assigned={st['summary']['assigned']} "
                      + " ".join(f"{r['node']}/{r['resource']}={','.join(r['owner']) or '-'}" for r in st["resources"]),
                      flush=True)
                last = key
        except Exception as e:
            print(f"[observer] {type(e).__name__}: {e}", flush=True)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
