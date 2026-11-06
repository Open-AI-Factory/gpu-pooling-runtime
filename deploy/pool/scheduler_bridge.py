import argparse
import sys
import time

from kubernetes import client, config

GROUP, VERSION, PLURAL = "grp.keti.re.kr", "v1alpha1", "planrequests"


def awaiting(co: client.CustomObjectsApi) -> list[dict]:
    items = co.list_cluster_custom_object(GROUP, VERSION, PLURAL)["items"]
    return [p for p in items if (p.get("status") or {}).get("result") == "AwaitingAllocation"
            and (p.get("status") or {}).get("allocationRequest")]


SKIPPED = "poolctl reserve() skipped by --skip-reserve: static MIG partitions are not reservable in poolctl v0.1"


def allocate(pr: dict, strategy: str, skip_reserve: bool = False) -> dict | None:
    from poolctl.adapters import MockSchedulerAdapter
    from poolctl.interface import PoolManager
    from poolctl.interface.wire import request_from_dict, reservation_to_dict
    from poolctl.sources import collect_from_resourceslices
    pm = PoolManager(collect_from_resourceslices())
    resp = pm.evaluate(request_from_dict(pr["status"]["allocationRequest"]))
    sel = MockSchedulerAdapter(strategy).select(resp)
    if sel is None:
        print(f"[bridge] {pr['metadata']['name']}: no candidate ({resp.diagnostics})", flush=True)
        return None
    if skip_reserve:
        from poolctl.interface.types import ReservationResult
        cand = next(c for c in resp.candidates if c.candidate_id == sel.selected_candidate_id)
        rr = ReservationResult(sel.request_id, sel.selected_candidate_id, True, SKIPPED,
                               tuple((p.node_id, s.gpu_id, s.profile, s.placement) for p in cand.pods for s in p.slots))
    else:
        rr = pm.reserve(sel)
    print(f"[bridge] {pr['metadata']['name']}: {resp.candidate_count} candidate(s), selected {sel.selected_candidate_id} "
          f"by {sel.decided_by}, reserved ok={rr.ok} {rr.reason or ''}", flush=True)
    return reservation_to_dict(rr) if rr.ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poolctl", required=True)
    ap.add_argument("--strategy", default="min_frag", choices=["first", "random", "min_frag", "min_link"])
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--skip-reserve", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, a.poolctl)
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    co = client.CustomObjectsApi()
    done: set[str] = set()
    while True:
        for pr in awaiting(co):
            md = pr["metadata"]
            if md["uid"] in done:
                continue
            done.add(md["uid"])
            try:
                rv = allocate(pr, a.strategy, a.skip_reserve)
            except Exception as e:
                print(f"[bridge] {md['name']}: {type(e).__name__}: {e}", flush=True)
                continue
            if rv:
                co.patch_namespaced_custom_object(GROUP, VERSION, md["namespace"], PLURAL, md["name"],
                                                  {"spec": {"allocation": {"source": "scheduler", "reservation": rv}}})
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
