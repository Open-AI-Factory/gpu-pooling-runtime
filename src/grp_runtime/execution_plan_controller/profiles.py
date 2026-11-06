from __future__ import annotations

import json
import math
from typing import Any, Callable

from ..pool import Gpu

SETTINGS_PROTOCOL = "runtime-plan-env-v1"
PROFILE_KEY = "profiles.json"
MAX_PROFILES = 128
ConfigMapReader = Callable[[str, str], dict]


class ProfileError(ValueError):
    pass


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ProfileError(msg)


def _int(v: Any, path: str, low: int = 1) -> None:
    _need(type(v) is int and v >= low, f"{path} must be an integer >= {low}")


def _num(v: Any, path: str) -> None:
    _need(type(v) in (int, float) and math.isfinite(v) and v >= 0, f"{path} must be a finite number >= 0")


def _obj(v: Any, path: str, keys: tuple[str, ...]) -> dict:
    _need(isinstance(v, dict), f"{path} must be an object")
    miss = sorted(set(keys) - set(v))
    _need(not miss, f"{path} missing {', '.join(miss)}")
    return v


def validate_profiles(profiles: Any) -> list[dict]:
    _need(isinstance(profiles, list), "profiles must be a list")
    _need(len(profiles) <= MAX_PROFILES, f"at most {MAX_PROFILES} profiles")
    seen: set[str] = set()
    for i, p in enumerate(profiles):
        at = f"profiles[{i}]"
        _obj(p, at, ("id", "kind", "runtime", "workload", "hardware", "config", "metrics", "evidence"))
        _need(isinstance(p["id"], str) and p["id"] and p["id"] not in seen, f"{at}.id must be a unique string")
        seen.add(p["id"])
        _obj(p["runtime"], f"{at}.runtime", ("name",))
        _int(_obj(p["workload"], f"{at}.workload", ("effectiveBatch",))["effectiveBatch"], f"{at}.workload.effectiveBatch")
        hw = _obj(p["hardware"], f"{at}.hardware", ("model", "mode", "migProfile"))
        _need(hw["mode"] in ("whole", "mig"), f"{at}.hardware.mode must be whole or mig")
        _need((hw["mode"] == "mig") == bool(hw["migProfile"]), f"{at}.hardware.migProfile must match mode")
        cfg = _obj(p["config"], f"{at}.config", ("batch", "parallelism", "placement", "memoryBudgetBytesPerDevice"))
        for k in ("microBatch", "gradientAccumulation"):
            _int(_obj(cfg["batch"], f"{at}.config.batch", ("microBatch", "gradientAccumulation"))[k], f"{at}.{k}")
        for k in ("dp", "tp", "pp"):
            _int(_obj(cfg["parallelism"], f"{at}.config.parallelism", ("dp", "tp", "pp"))[k], f"{at}.{k}")
        for k in ("nodeCount", "gpusPerNode"):
            _int(_obj(cfg["placement"], f"{at}.config.placement", ("nodeCount", "gpusPerNode"))[k], f"{at}.{k}")
        _int(cfg["memoryBudgetBytesPerDevice"], f"{at}.memoryBudgetBytesPerDevice")
        m = _obj(p["metrics"], f"{at}.metrics", ("throughput", "peakMemoryBytesPerDevice", "qualityPassed"))
        _num(m["throughput"], f"{at}.metrics.throughput")
        _int(m["peakMemoryBytesPerDevice"], f"{at}.metrics.peakMemoryBytesPerDevice", 0)
        _need(type(m["qualityPassed"]) is bool, f"{at}.metrics.qualityPassed must be boolean")
        _need(_obj(p["evidence"], f"{at}.evidence", ("kind", "ref"))["kind"] in ("measured", "synthetic"),
              f"{at}.evidence.kind must be measured or synthetic")
    return profiles


def _bundle_profiles(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get("acceptedProfiles", raw.get("profiles"))
    return raw


def _read_configmap(name: str, namespace: str) -> dict:
    from kubernetes import client
    cm = client.CoreV1Api().read_namespaced_config_map(name, namespace)
    return {"data": cm.data or {}, "resourceVersion": cm.metadata.resource_version}


def load_profiles(want: dict[str, Any], namespace: str | None,
                  reader: ConfigMapReader | None = None) -> tuple[list[dict], str]:
    if "profiles" in want:
        return validate_profiles(_bundle_profiles(want["profiles"])), "launch.settings.profiles"
    name = want.get("profileConfigMap")
    if not name:
        return [], ""
    _need(bool(namespace), "profileConfigMap needs a namespace")
    try:
        cm = (reader or _read_configmap)(name, namespace)
    except Exception as e:
        raise ProfileError(f"cannot read ConfigMap {namespace}/{name}: {getattr(e, 'reason', None) or e}") from e
    _need(PROFILE_KEY in cm["data"], f"ConfigMap {name} has no {PROFILE_KEY}")
    try:
        raw = json.loads(cm["data"][PROFILE_KEY])
    except ValueError as e:
        raise ProfileError(f"ConfigMap {name} {PROFILE_KEY} is not JSON") from e
    return validate_profiles(_bundle_profiles(raw)), f"configmap/{name}@{cm.get('resourceVersion', '')}"


def _model(name: str) -> str:
    n = " ".join((name or "").replace("-", " ").replace("_", " ").split()).lower()
    return n[len("nvidia "):] if n.startswith("nvidia ") else n


def hardware_of(g: Gpu) -> dict[str, Any]:
    mig = g.uuid.startswith("MIG-") or bool(g.mig_profile)
    return {"model": g.model, "mode": "mig" if mig else "whole", "migProfile": g.mig_profile if mig else "",
            "memoryBytes": int(g.memory_gb * 2 ** 30)}


def _same_hw(p: dict, hw: dict) -> bool:
    h = p["hardware"]
    return (_model(h["model"]) == _model(hw["model"]) and h["mode"] == hw["mode"]
            and (h["migProfile"] or "") == hw["migProfile"])


def _identity_reasons(p: dict, ident: dict[str, Any]) -> list[str]:
    out = []
    if "model" in ident and p.get("model") != ident["model"]:
        out.append("identity_mismatch:model")
    if "precision" in ident and p.get("precision") != ident["precision"]:
        out.append("identity_mismatch:precision")
    if "signature" in ident and (p.get("workload") or {}).get("signature") != ident["signature"]:
        out.append("identity_mismatch:signature")
    return out


def _situation(p: dict, total: int, par: dict, per_node: dict[str, int], hws: list[dict],
               ident: dict[str, Any]) -> list[str]:
    cfg, out = p["config"], []
    if p["kind"] != "training" or p["runtime"]["name"] != "pytorch":
        out.append("not_pytorch_training")
    out += _identity_reasons(p, ident)
    if p["workload"]["effectiveBatch"] != total:
        out.append("effective_batch_differs")
    if {k: cfg["parallelism"][k] for k in ("dp", "tp", "pp")} != {k: par[k] for k in ("dp", "tp", "pp")}:
        out.append("parallelism_differs")
    counts = set(per_node.values())
    if cfg["placement"]["nodeCount"] != len(per_node) or counts != {cfg["placement"]["gpusPerNode"]}:
        out.append("placement_differs")
    if not hws or not all(_same_hw(p, h) for h in hws):
        out.append("hardware_differs")
    return out


def _feasibility(p: dict, total: int, cap: int, dp: int, hws: list[dict], allow_synthetic: bool) -> list[str]:
    cfg, m, out = p["config"], p["metrics"], []
    b = cfg["batch"]
    if b["microBatch"] * b["gradientAccumulation"] * dp != total:
        out.append("effective_batch_changed")
    if b["microBatch"] > cap:
        out.append("micro_batch_over_cap")
    if not m["qualityPassed"]:
        out.append("quality_failed")
    if m["peakMemoryBytesPerDevice"] > cfg["memoryBudgetBytesPerDevice"]:
        out.append("budget_below_peak")
    if any(h["memoryBytes"] and cfg["memoryBudgetBytesPerDevice"] > h["memoryBytes"] for h in hws):
        out.append("budget_over_device_memory")
    if p["evidence"]["kind"] == "synthetic" and not allow_synthetic:
        out.append("synthetic_evidence_disabled")
    return out


def rule_settings(total: int, cap: int, dp: int) -> dict[str, Any]:
    per_rank = total // dp
    micro = max(m for m in range(1, min(cap, per_rank) + 1) if per_rank % m == 0)
    return {"protocol": SETTINGS_PROTOCOL, "effectiveBatch": total, "source": "rule",
            "batch": {"microBatch": micro, "gradientAccumulation": per_rank // micro},
            "rule": "preserve effective batch; micro-batch capped by maxMicroBatch"}


def brief(settings: dict[str, Any] | None) -> dict[str, Any] | None:
    if not settings:
        return None
    return {"source": settings.get("source"), "batch": settings.get("batch"),
            "profileId": (settings.get("profiles") or {}).get("profileId")}


def select(launch: dict[str, Any] | None, par: dict[str, int], devices: list[Gpu] | None = None,
           namespace: str | None = None, reader: ConfigMapReader | None = None) -> tuple[dict[str, Any] | None, str]:
    want = (launch or {}).get("settings")
    if not want:
        return None, ""
    total, cap, dp = int(want.get("effectiveBatch", 0)), int(want.get("maxMicroBatch", 0)), par["dp"]
    if total < 1 or cap < 1:
        return None, "launch.settings needs effectiveBatch and maxMicroBatch >= 1"
    if total % dp:
        return None, f"effectiveBatch {total} is not divisible by DP={dp}"
    pin = want.get("batch")
    if pin:
        micro, accum = int(pin.get("microBatch", 0)), int(pin.get("gradientAccumulation", 0))
        if micro < 1 or accum < 1 or micro * accum * dp != total:
            return None, f"launch.settings.batch {pin} x DP={dp} does not give effectiveBatch {total}"
        return {"protocol": SETTINGS_PROTOCOL, "effectiveBatch": total, "source": "pinned",
                "batch": {"microBatch": micro, "gradientAccumulation": accum},
                "rule": "batch fixed by launch.settings.batch"}, ""
    try:
        profiles, origin = load_profiles(want, namespace, reader)
    except ProfileError as e:
        return None, f"launch.settings profiles: {e}"
    rule = rule_settings(total, cap, dp)
    if not profiles:
        return rule, ""
    devices = devices or []
    hws = [hardware_of(g) for g in devices]
    per_node: dict[str, int] = {}
    for g in devices:
        per_node[g.node] = per_node.get(g.node, 0) + 1
    ident = want.get("identity") or {}
    evaluations, feasible = [], []
    for p in profiles:
        reasons = _situation(p, total, par, per_node, hws, ident)
        matched = not reasons
        if matched:
            reasons = _feasibility(p, total, cap, dp, hws, bool(want.get("allowSynthetic", False)))
            if not reasons:
                feasible.append(p)
        evaluations.append({"profileId": p["id"], "matched": matched, "reasons": reasons})
    where = (f"DP{par['dp']} TP{par['tp']} PP{par['pp']} on "
             + ", ".join(sorted({f"{h['model'] or '?'}{'/' + h['migProfile'] if h['migProfile'] else ''}"
                                 for h in hws}) or ["no devices"]))
    matched = [e for e in evaluations if e["matched"]]
    base = {"origin": origin, "evaluated": len(profiles), "matched": len(matched)}
    if not matched:
        return {**rule, "profiles": {**base, "status": "needsProfiling",
                                     "reason": f"no profile measured for {where}"}}, ""
    if not feasible:
        why = "; ".join(f"{e['profileId']}: {','.join(e['reasons'])}" for e in matched[:5])
        return None, f"noFeasiblePlan: {len(matched)} profile(s) measured for {where}, none usable ({why})"
    best = min(feasible, key=lambda p: (-p["metrics"]["throughput"], p["id"]))
    cfg = best["config"]
    return {"protocol": SETTINGS_PROTOCOL, "effectiveBatch": total, "source": "profile",
            "batch": {"microBatch": cfg["batch"]["microBatch"],
                      "gradientAccumulation": cfg["batch"]["gradientAccumulation"]},
            "memoryBudgetBytesPerDevice": cfg["memoryBudgetBytesPerDevice"],
            "profiles": {**base, "status": "planned", "profileId": best["id"],
                         "evidence": dict(best["evidence"]), "throughput": best["metrics"]["throughput"],
                         "feasible": len(feasible),
                         "reason": f"best measured throughput among {len(feasible)} feasible profile(s) for {where}"}}, ""
