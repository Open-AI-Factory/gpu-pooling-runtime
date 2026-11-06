from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .models import ACTIONS, KEEP_LAYOUT_ACTIONS

API = "grp.keti.re.kr/v1alpha1"
POOL = "k8s:gpuresourcepools/gpu-pool"
MODELS = {
    "ddp": {"runtime": "ddp", "script": "workloads/train_ddp.py",
            "profiling": {"weightsGb": 0.0196, "gradientsGb": 0.0196, "optimizerGb": 0.0391,
                          "activationGbPerSample": 3.5e-05}},
    "3d": {"runtime": "pytorch-3d", "script": "workloads/train_3d.py",
           "profiling": {"weightsGb": 0.1, "gradientsGb": 0.1, "optimizerGb": 0.2, "activationGbPerSample": 0.0001}},
}
LAYOUT_ACTIONS = tuple(a for a in ACTIONS if a not in KEEP_LAYOUT_ACTIONS)


class SubmitError(ValueError):
    pass


def _ids(text: str) -> list[str]:
    ids = [x.strip() for x in (text or "").replace("\n", ",").split(",") if x.strip()]
    if len(set(ids)) != len(ids):
        raise SubmitError(f"duplicate GPU ids: {ids}")
    return ids


def _par(dp: int, tp: int, pp: int) -> dict[str, int]:
    if min(dp, tp, pp) < 1:
        raise SubmitError("dp, tp and pp must be >= 1")
    return {"dp": dp, "tp": tp, "pp": pp}


def plan_request(name: str, model: str, dp: int, tp: int, pp: int, gpu_ids: str = "", image: str = "",
                 max_steps: int = 3000, step_sleep: float = 0.05, effective_batch: int = 1024,
                 max_micro_batch: int = 256, profile_configmap: str = "", checkpoint_claim: str = "grp-checkpoints",
                 mlflow_uri: str = "", source: str = "argo", request_id: str = "") -> dict[str, Any]:
    if model not in MODELS:
        raise SubmitError(f"model must be one of {sorted(MODELS)}")
    if not image:
        raise SubmitError("image is required")
    m, par, ids = MODELS[model], _par(dp, tp, pp), _ids(gpu_ids)
    world = dp * tp * pp
    if model == "ddp" and (tp, pp) != (1, 1):
        raise SubmitError("model ddp runs data parallel only; use model 3d for TP or PP")
    if ids and len(ids) != world:
        raise SubmitError(f"{len(ids)} GPU id(s) given but DP×TP×PP = {world}")
    if effective_batch % dp:
        raise SubmitError(f"effective batch {effective_batch} is not divisible by DP={dp}")
    args = ["--max-steps", str(max_steps), "--global-batch", str(effective_batch)]
    if model == "ddp":
        args += ["--step-sleep", str(step_sleep)]
    env = {"PYTHONUNBUFFERED": "1", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"}
    if mlflow_uri:
        env.update(MLFLOW_TRACKING_URI=mlflow_uri, MLFLOW_EXPERIMENT_NAME="grp-runtime", GRP_RUN_NAME=name)
    launch: dict[str, Any] = {"script": m["script"], "args": args, "cpu": "1", "memory": "4Gi", "gpu": 1,
                              "runtimeClass": "nvidia", "image": image, "python": "python", "codeInImage": True,
                              "checkpoint": {"claimName": checkpoint_claim}, "env": env}
    if model == "ddp":
        launch["settings"] = {"effectiveBatch": effective_batch, "maxMicroBatch": max_micro_batch,
                              **({"profileConfigMap": profile_configmap} if profile_configmap else {})}
    spec: dict[str, Any] = {"requestId": request_id or name, "source": source, "workloadRef": name,
                            "runtime": m["runtime"], "parallelism": par,
                            "profiling": {"workloadType": "training", **m["profiling"], "globalBatch": effective_batch},
                            "poolSnapshot": POOL, "launch": launch}
    if ids:
        spec["allocation"] = {"gpuIds": ids}
    return {"apiVersion": API, "kind": "PlanRequest", "metadata": {"name": name}, "spec": spec}


def runtime_action(name: str, plan: str, action: str, dp: int = 0, tp: int = 0, pp: int = 0, gpu_ids: str = "",
                   profile_configmap: str = "", max_micro_batch: int = 0, micro_batch: int = 0,
                   gradient_accumulation: int = 0, timeout: int = 120) -> dict[str, Any]:
    if not plan:
        raise SubmitError("plan (the training ExecutionPlan name) is required")
    if action not in ACTIONS:
        raise SubmitError(f"action must be one of {list(ACTIONS)}")
    target: dict[str, Any] = {}
    if action in LAYOUT_ACTIONS:
        par, ids = _par(dp, tp, pp), _ids(gpu_ids)
        target = {"gpus": dp * tp * pp, "parallelism": par}
        if ids:
            if len(ids) != target["gpus"]:
                raise SubmitError(f"{len(ids)} GPU id(s) given but DP×TP×PP = {target['gpus']}")
            target["allocation"] = {"gpuIds": ids}
    elif action == "ApplySettings":
        settings: dict[str, Any] = {}
        if profile_configmap:
            settings["profileConfigMap"] = profile_configmap
        if max_micro_batch:
            settings["maxMicroBatch"] = max_micro_batch
        if micro_batch or gradient_accumulation:
            if micro_batch < 1 or gradient_accumulation < 1:
                raise SubmitError("micro batch and gradient accumulation must be given together")
            settings["batch"] = {"microBatch": micro_batch, "gradientAccumulation": gradient_accumulation}
        if not settings:
            raise SubmitError("ApplySettings needs a profile ConfigMap, a max micro batch or a fixed batch")
        target = {"settings": settings}
    spec: dict[str, Any] = {"executionPlanRef": plan, "action": action, "target": target}
    if action != "Observe":
        spec["safePolicy"] = {"mode": "WaitSafePoint", "timeoutSeconds": timeout}
    return {"apiVersion": API, "kind": "RuntimeAction", "metadata": {"name": name}, "spec": spec}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grp_runtime.submit")
    sub = ap.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="ddp")
    p.add_argument("--image", required=True)
    for k in ("dp", "tp", "pp"):
        p.add_argument(f"--{k}", type=int, default=1)
    p.add_argument("--gpu-ids", default="")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--step-sleep", type=float, default=0.05)
    p.add_argument("--effective-batch", type=int, default=1024)
    p.add_argument("--max-micro-batch", type=int, default=256)
    p.add_argument("--profile-configmap", default="")
    p.add_argument("--checkpoint-claim", default="grp-checkpoints")
    p.add_argument("--mlflow-uri", default="")
    p.add_argument("--request-id", default="")
    a = sub.add_parser("action")
    a.add_argument("--name", required=True)
    a.add_argument("--plan", required=True)
    a.add_argument("--action", required=True)
    for k in ("dp", "tp", "pp"):
        a.add_argument(f"--{k}", type=int, default=0)
    a.add_argument("--gpu-ids", default="")
    a.add_argument("--profile-configmap", default="")
    a.add_argument("--max-micro-batch", type=int, default=0)
    a.add_argument("--micro-batch", type=int, default=0)
    a.add_argument("--gradient-accumulation", type=int, default=0)
    a.add_argument("--timeout", type=int, default=120)
    ns = vars(ap.parse_args(argv))
    kind = ns.pop("kind")
    try:
        doc = plan_request(**ns) if kind == "plan" else runtime_action(**ns)
    except SubmitError as e:
        print(f"invalid request: {e}", file=sys.stderr)
        return 2
    print(json.dumps(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
