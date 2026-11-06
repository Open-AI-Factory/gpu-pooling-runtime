from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml
from kubernetes import client, config, watch

from .runtime_adapter.workload import Workload

GROUP, VERSION = "grp.keti.re.kr", "v1alpha1"
DONE_RESULTS = ("request validated", "preparation=Ready", "serviceConfig=Valid", "restart=Running",
                "ready=true response=OK", "queued")


def _co():
    config.load_kube_config()
    return client.CustomObjectsApi()


def cmd_agents(a) -> int:
    co = _co()
    eps = co.list_namespaced_custom_object(GROUP, VERSION, a.namespace, "executionplans")["items"]
    ep = next((e for e in eps if e["spec"]["workloadRef"] == a.workload), None)
    if ep is None:
        print(f"no ExecutionPlan for workload {a.workload}")
        return 1
    wl = Workload(a.namespace, a.workload)
    requested = ((ep.get("status") or {}).get("isolation") or {}).get("requested", [])
    r = wl.readiness(requested)
    print(f"agents hook={r['hook']}  ranks={','.join(map(str, r['ranks'])) or '-'}  world={r['world']}")
    print(f"agent isolation={r['isolation']}  state={r['state']}")
    if a.verbose or r["state"] != "ReadyToRun":
        agents = {x["rank"]: x for x in wl.agents()}
        for req in requested:
            ag = agents.get(req["rank"], {})
            iso = ag.get("isolation", {})
            print(f"  rank{req['rank']}  pod={ag.get('pod')}  hook={ag.get('hook')}  "
                  f"node {req['node']}/{ag.get('node')}  cpu {req['cpuLimit']}/{iso.get('cpuLimit')}  "
                  f"mem {req['memoryLimitMb']}Mi/{iso.get('memoryLimitMb')}Mi  "
                  f"device {req['device']}/{iso.get('device')}   (requested/actual)")
        for m in r["mismatches"]:
            print("  mismatch:", m)
        if r["staleRanks"]:
            print("  unresponsive ranks:", r["staleRanks"])
    return 0


_last = {"name": None}


def _print_new(name: str, obj: dict, seen: dict, verbose: bool) -> None:
    st = obj.get("status") or {}
    conds = st.get("conditions") or []
    if name not in seen:
        tgt = (obj["spec"].get("target") or {})
        print(f"action={name}  mode={obj['spec'].get('mode', 'restart')}  "
              f"target.gpus={tgt.get('gpus', tgt.get('gpuCount'))}", flush=True)
        seen[name], _last["name"] = 0, name
    for c in conds[seen[name]:]:
        if c["phase"] == "Aborted":
            print(f"action={name}  Aborted", flush=True)
            print(f"  {c['result'].replace(' ', '  ')}", flush=True)
            _last["name"] = name
        elif verbose or c["result"] in DONE_RESULTS:
            if _last["name"] != name:
                print(f"action={name}", flush=True)
                _last["name"] = name
            print(f"  {c['phase']:<12} {c['result']}" + (f"   ({c['message']})" if verbose and c['message'] else ""),
                  flush=True)
    seen[name] = len(conds)


def cmd_watch(a) -> int:
    co = _co()
    seen: dict[str, int] = {}
    start = time.time()
    items = co.list_namespaced_custom_object(GROUP, VERSION, a.namespace, "runtimeactions")["items"]
    for o in sorted(items, key=lambda o: o["metadata"]["creationTimestamp"]):
        if a.all:
            _print_new(o["metadata"]["name"], o, seen, a.verbose)
        else:
            seen[o["metadata"]["name"]] = len((o.get("status") or {}).get("conditions") or [])
    w = watch.Watch()
    try:
        for ev in w.stream(co.list_namespaced_custom_object, GROUP, VERSION, a.namespace, "runtimeactions",
                           timeout_seconds=a.timeout or None):
            o = ev["object"]
            _print_new(o["metadata"]["name"], o, seen, a.verbose)
            if a.timeout and time.time() - start > a.timeout:
                break
    except KeyboardInterrupt:
        pass
    return 0


def _print_candidates(res) -> None:
    for r in (res.decision or {}).get("candidates") or []:
        print(f"             DP{r['dp']} TP{r['tp']} PP{r['pp']}  {'O' if r['feasible'] else 'X'}  {r['note']}"
              + ("  <- selected" if r.get("selected") else ""))


def _finish(res, a) -> int:
    from .execution_plan_controller import intake
    if res.result == "Generated" and a.apply:
        res = intake.register(res, allow_stale_pool=getattr(a, "allow_stale_pool", False))
        print(f"[register]   {res.result}" + (f" reason={res.reason}" if res.reason else "") + f"  {res.message}")
        if res.result == "Registered" and a.wait:
            v = intake.wait_validation(res.registration["name"], res.registration["namespace"], a.wait)
            res.registration["controller"] = v
            print(f"[controller] " + (f"validation={v['validation']} assignedNodes={v.get('assignedNodes')}  {v['reason']}"
                                      if v else f"no status.validation within {a.wait}s (is the operator running?)"))
    elif res.result == "Generated":
        print("[register]   skipped (add --apply to submit to the cluster)")
    d = intake.record(res, Path(a.runs_dir)) if getattr(a, "runs_dir", None) else None
    if d:
        print(f"[record]     {d}")
    print(f"result={res.result}" + (f" reason={res.reason}" if res.reason else "") + (f"  {res.message}" if res.result in ("Rejected", "Duplicate") else ""))
    for e in res.errors:
        print(f"  {e['field']}: {e['code']} - {e['message']}")
    if res.manifest and a.output:
        print(yaml.safe_dump(res.manifest, allow_unicode=True, sort_keys=False))
    return 1 if res.result == "Rejected" else 0


def _print_decision(res) -> None:
    d = res.decision or {}
    p = d.get("pool")
    if p:
        print(f"[pool]       source={p['source']} version={p['version']} atomicSnapshot={str(p['atomicSnapshot']).lower()} "
              f"freeGPUs={p['freeGpus']} excluded={len(p['excluded'])}")
        for x in p["excluded"]:
            print(f"             excluded {x['device']} ({x['model']}): {x['reason']}")
    if d.get("topology"):
        t = d["topology"]
        print(f"[topology]   usable GPUs per node={t['usableGpusPerNode']} memory={t.get('memoryGb', '-')}GB/GPU "
              f"interconnect={t.get('interconnect')}")
    if d.get("mode") == "Given":
        s = d.get("selected") or {}
        print(f"[request]    parallelism given: DP{s.get('dp')} TP{s.get('tp')} PP{s.get('pp')} "
              f"-> {d['gpuCount']['value']} GPU(s)")
        for c in d.get("placementCandidates") or []:
            print(f"[placement]  {c['label']}{' *' if c['selected'] else '  '} TP {c['tpGroups']}, PP {c['ppStages']}, "
                  f"cross-node TP/DP/PP {c['tpCrossNode']}/{c['dpCrossNode']}/{c['ppCrossNode']}, "
                  f"nodes {c['nodesUsed']}, fragmented {c['fragmentedNodes']}")
            for n, cells in c["layout"].items():
                print(f"               {n}: {' '.join(cells)}")
    elif d.get("constraint"):
        print(f"[constraint] pinned GPUs={d['constraint']['gpuIds']} (placement must use these)")
    if d.get("workers") and d.get("mode") != "Given":
        w = d["workers"]
        print(f"[workers]    GPUs={w['gpus']} (request, fixed) -> {w['nodes']} node slot(s) x {w['gpusPerNode']} GPU "
              f"(gpusPerNode from {w['gpusPerNodeSource']})")
    if d.get("candidates"):
        print(f"[decide]     mode={d.get('mode')} candidates={len(d['candidates'])}")
        _print_candidates(res)
    if d.get("selected") and res.manifest:
        s = d["selected"]
        print(f"[generate]   ExecutionPlan {res.manifest['metadata']['name']}  DP{s['dp']} TP{s['tp']} PP{s['pp']}  "
              f"({d['reason']})")
        if d.get("rankMap"):
            for x in d["rankMap"]:
                print(f"               rank{x['rank']} DP{x['dp'] + 1} PP{x['pp'] + 1} TP{x['tp'] + 1} -> {x['gpu']}")
            ex = d.get("execution") or {}
            print(f"[execution]  executable={str(ex.get('executable')).lower()} ({ex.get('reason')})")
        else:
            w = d["workers"]
            print(f"[to sched]   world={w['world']} rank->node slot {[x['nodeSlot'] for x in w['rankSlots']]} "
                  f"constraints={w['constraints']}  (node names are decided by the scheduler)")
    for o in (d.get("feedback") or {}).get("options") or []:
        print(f"[feedback]   DP{o['dp']} TP{o['tp']} PP{o['pp']} needs >= {o['minGpusPerNode']} GPU/node, "
              f">= {o['minGpuMemoryGb']}GB per GPU")
    for w in d.get("warnings") or []:
        print(f"  warning: {w}")


def cmd_plan(a) -> int:
    from .execution_plan_controller import intake
    prof_doc = yaml.safe_load(open(a.profiling, encoding="utf-8")) if a.profiling else {}
    doc = {"kind": "PlanRequest", "metadata": {"name": a.name, "namespace": a.namespace},
           "spec": {"requestId": a.name, "source": "keticti-cli", "workloadRef": a.workload, "runtime": a.runtime,
                    "resources": {"gpus": a.gpus, "gpusPerNode": a.gpus_per_node},
                    "interconnect": a.interconnect, "profiling": prof_doc, "poolSnapshot": a.pool,
                    "launch": {"script": "workloads/train_ddp.py", "args": a.args.split() if a.args else [],
                               "cpu": a.cpu, "memory": a.memory}}}
    res = intake.generate(doc)
    _print_decision(res)
    a.runs_dir, a.output = None, not a.apply
    return _finish(res, a)


def cmd_plan_request(a) -> int:
    from .execution_plan_controller import intake
    doc = intake.load_request(a.request)
    s = doc.get("spec") if isinstance(doc, dict) else None
    s = s if isinstance(s, dict) else {}
    print(f"[receive]    request={s.get('requestId')} source={s.get('source', 'file')} "
          f"workload={s.get('workloadRef')} runtime={s.get('runtime')} resources={s.get('resources')}"
          + (f" parallelism={s.get('parallelism')}" if s.get("parallelism") else ""))
    prev = intake.previous(doc, Path(a.runs_dir))
    if prev and prev.result == "Rejected":
        print("[validate]   requestId already used with different content")
        a.runs_dir = None
        return _finish(prev, a)
    if prev and (prev.result == "Registered" or not a.apply):
        print(f"[validate]   same request already processed at {prev.time} ({prev.result}) - not re-decided")
        prev.result, prev.reason = "Duplicate", "DuplicateRequest"
        prev.message = f"previous result kept: {Path(a.runs_dir) / prev.request_id}"
        a.runs_dir = None
        return _finish(prev, a)
    if prev:
        print(f"[validate]   generated earlier at {prev.time}; registering the recorded plan")
        res = prev
    else:
        res = intake.generate(doc, allow_stale_pool=a.allow_stale_pool)
        print("[validate]   " + ("OK" if res.reason != "InvalidSpec" else f"{len(res.errors)} error(s)"))
    _print_decision(res)
    return _finish(res, a)


def cmd_quota(a) -> int:
    co = _co()
    eps = co.list_namespaced_custom_object(GROUP, VERSION, a.namespace, "executionplans")["items"]
    ep = next((e for e in eps if e["spec"]["workloadRef"] == a.workload), None)
    if ep is None:
        print(f"no ExecutionPlan for workload {a.workload}")
        return 1
    ras = co.list_namespaced_custom_object(GROUP, VERSION, a.namespace, "runtimeactions")["items"]
    n = 1 + max([int(r["metadata"]["name"].split("-")[-1]) for r in ras
                 if r["metadata"]["name"].startswith("resize-") and r["metadata"]["name"].split("-")[-1].isdigit()]
                or [0])
    name = a.name or f"resize-{n:02d}"
    body = {"apiVersion": f"{GROUP}/{VERSION}", "kind": "RuntimeAction",
            "metadata": {"name": name, "labels": {"grp.keti.re.kr/trigger": "quota",
                                                  "grp.keti.re.kr/workload": a.workload}},
            "spec": {"executionPlanRef": ep["metadata"]["name"], "action": "Resize", "mode": "restart",
                     "target": {"gpus": a.gpus},
                     "safePolicy": {"mode": "WaitSafePoint", "timeoutSeconds": a.deadline}}}
    if a.inject_failure:
        body["spec"]["adapterOptions"] = {"failAt": "Reconfigure"}
    co.create_namespaced_custom_object(GROUP, VERSION, a.namespace, "runtimeactions", body)
    print(f"quota {a.workload}: GPUs -> {a.gpus}  =>  RuntimeAction {name} created (preemption request, preparation deadline {a.deadline}s)")
    return 0


def cmd_debug(a) -> int:
    wl = Workload(a.namespace, a.workload, core=(config.load_kube_config() or client.CoreV1Api()))
    ag = next((x for x in wl.agents() if x["rank"] == a.rank), None)
    if not ag:
        print(f"no Agent for rank {a.rank}")
        return 1
    sig = "STOP" if a.op == "pause" else "CONT"
    subprocess.run(["kubectl", "exec", "-n", a.namespace, ag["pod"], "--", "sh", "-c",
                    f"kill -{sig} {ag['pid']}"], check=True)
    print(f"rank {a.rank} ({ag['pod']} pid {ag['pid']}) {'paused (simulating unresponsive rank)' if sig == 'STOP' else 'resumed'}")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="keticti")
    ap.add_argument("-n", "--namespace", default="default")
    top = ap.add_subparsers(dest="group", required=True)

    rt = top.add_parser("runtime").add_subparsers(dest="cmd", required=True)
    p = rt.add_parser("agents")
    p.add_argument("workload")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_agents)
    p = rt.add_parser("watch")
    p.add_argument("--all", action="store_true", help="also print existing actions from the beginning")
    p.add_argument("-v", "--verbose", action="store_true", help="include intermediate steps and messages")
    p.add_argument("--timeout", type=int, default=0)
    p.set_defaults(fn=cmd_watch)

    pl = top.add_parser("plan").add_subparsers(dest="cmd", required=True)
    p = pl.add_parser("create")
    p.add_argument("name")
    p.add_argument("--workload", required=True)
    p.add_argument("--runtime", default="fsdp", choices=["ddp", "fsdp"])
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--interconnect", default="rdma")
    p.add_argument("--profiling", help="profiling YAML path")
    p.add_argument("--pool", default="mock/kind-snapshot.yaml")
    p.add_argument("--cpu", default="1")
    p.add_argument("--memory", default="1Gi")
    p.add_argument("--args", default="", help="training script arguments")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--wait", type=float, default=0, help="seconds to wait for status.validation after --apply")
    p.set_defaults(fn=cmd_plan)
    p = pl.add_parser("request", help="request file -> validate -> pool -> decide -> ExecutionPlan [-> register]")
    p.add_argument("request", help="PlanRequest YAML path")
    p.add_argument("--apply", action="store_true", help="register the generated ExecutionPlan (kubectl create)")
    p.add_argument("--wait", type=float, default=0, help="seconds to wait for status.validation after --apply")
    p.add_argument("--allow-stale-pool", action="store_true",
                   help="accept a pool status past validUntil (offline copy); recorded as a warning")
    p.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[2] / "runs" / "plans"))
    p.add_argument("-o", "--output", action="store_true", help="print the generated ExecutionPlan YAML")
    p.set_defaults(fn=cmd_plan_request)

    q = top.add_parser("quota").add_subparsers(dest="cmd", required=True)
    p = q.add_parser("set")
    p.add_argument("workload")
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--deadline", type=float, default=60, help="preparation deadline before change (seconds)")
    p.add_argument("--name")
    p.add_argument("--inject-failure", action="store_true", help="(demo) inject a failure in the restart step")
    p.set_defaults(fn=cmd_quota)

    d = top.add_parser("debug").add_subparsers(dest="op", required=True)
    for op in ("pause", "resume"):
        p = d.add_parser(op)
        p.add_argument("workload")
        p.add_argument("--rank", type=int, default=1)
        p.set_defaults(fn=cmd_debug, op=op)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
