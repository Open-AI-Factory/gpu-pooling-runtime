from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from .runtime_coordination.runtime_action_controller.controller import RuntimeActionController
from .models import RuntimeActionSpec
from .execution_plan_controller.planner import build_plan
from .pool import MockPoolClient

ROOT = Path(__file__).resolve().parents[2]


def load(path: str) -> RuntimeActionSpec:
    p = Path(path)
    spec = RuntimeActionSpec.from_manifest(yaml.safe_load(p.read_text(encoding="utf-8")))
    if spec.pool_snapshot and not Path(spec.pool_snapshot).is_absolute():
        spec.pool_snapshot = str((ROOT / spec.pool_snapshot).resolve())
    return spec


def cmd_run(a) -> int:
    spec = load(a.manifest)
    out = Path(a.runs_dir) / spec.name
    out.mkdir(parents=True, exist_ok=True)
    events = (out / "events.jsonl").open("w", encoding="utf-8")

    def on_transition(st):
        c = st.conditions[-1]
        events.write(json.dumps({**c, "status": st.to_dict() | {"conditions": None}},
                                ensure_ascii=False) + "\n")
        events.flush()

    st = RuntimeActionController(Path(a.runs_dir), on_transition).run(spec)
    events.close()
    doc = {"apiVersion": "grp.keti.re.kr/v1alpha1", "kind": "RuntimeAction",
           "metadata": {"name": spec.name}, "status": st.to_dict()}
    if not a.verbose and doc["status"]["executionPlan"]:
        doc["status"]["executionPlan"]["candidates"] = (
            f"{len(doc['status']['executionPlan']['candidates'])} (see `plan` command or status.yaml)")
    text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    (out / "status.yaml").write_text(
        yaml.safe_dump({**doc, "status": st.to_dict()}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print("\n" + text)
    print(f"-> saved: {out}")
    return 0 if st.phase.value == "Succeeded" else 1


def cmd_plan(a) -> int:
    spec = load(a.manifest)
    pool = MockPoolClient(spec.pool_snapshot)
    cur = [g.id for g in pool.gpus_of(spec.workload_ref.key)]
    cs = pool.get_candidates(spec.workload_ref.key, cur, spec.target_gpu_count)
    print(f"snapshot={cs.snapshot_version} workload={spec.workload_ref.key} currentGPUs={cur} "
          f"target={spec.target_gpu_count}")
    for c in cs.candidates:
        print(f"\ncandidate#{c.rank} {c.gpu_ids} ({c.interconnect}, {c.memory_gb}GB) — {c.reason}")
        plan = build_plan(spec.target_gpu_count, spec.profiling, c, "-")
        rows = plan.candidates if plan else []
        if not plan:
            from .execution_plan_controller.planner import enumerate_configs, evaluate
            for dp, tp, pp in enumerate_configs(spec.target_gpu_count):
                ok, why, mem = evaluate(dp, tp, pp, spec.profiling, c)
                rows.append({"dp": dp, "tp": tp, "pp": pp, "feasible": ok, "note": why})
        print(f"  {'DP':>3} {'TP':>3} {'PP':>3}  ok    note")
        for r in rows:
            mark = "<- selected" if r.get("selected") else ""
            print(f"  {r['dp']:>3} {r['tp']:>3} {r['pp']:>3}  {'O' if r['feasible'] else 'X':^4}  "
                  f"{r['note']} {mark}")
        if plan:
            print(f"  → {plan.reason}")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="grp_runtime")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one RuntimeAction")
    r.add_argument("manifest")
    r.add_argument("-v", "--verbose", action="store_true", help="print all DP/TP/PP candidates")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("plan", help="print ExecutionPlan candidates and selection reason only")
    p.add_argument("manifest")
    p.set_defaults(fn=cmd_plan)
    from .recovery.cli import add_parsers
    add_parsers(sub)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    return a.fn(a)
