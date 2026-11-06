import pytest

from grp_runtime.models import Profiling
from grp_runtime.execution_plan_controller.planner import build_plan, evaluate
from grp_runtime.pool import Candidate


def prof(**kw):
    base = dict(weights_gb=2.0, gradients_gb=2.0, optimizer_gb=4.0,
                activation_gb_per_sample=0.01, global_batch=1024)
    return Profiling(**{**base, **kw})


def cand(n=2, mem=80.0, share=1):
    return Candidate([f"a:{i}" for i in range(n)], ["a"], "NVLink", n, mem, ranks_per_physical=share)


def test_activation_grows_when_dp_shrinks():
    p = prof()
    _, _, m2 = evaluate(2, 1, 1, p, cand(2))
    _, _, m1 = evaluate(1, 1, 1, p, cand(1))
    assert m2 == pytest.approx(8 + 5.12) and m1 == pytest.approx(8 + 10.24)


def test_scaledown_rejected_when_activation_exceeds_memory():
    p = prof(activation_gb_per_sample=0.06)
    ok2, *_ = evaluate(2, 1, 1, p, cand(2, mem=40))
    ok1, why, _ = evaluate(1, 1, 1, p, cand(1, mem=40))
    assert not ok1 and "per-rank batch 1024" in why


def test_shared_physical_gpu_sums_ranks():
    p = prof(weights_gb=1, gradients_gb=1, optimizer_gb=2, activation_gb_per_sample=0.001)
    ok_alone, *_ = evaluate(2, 1, 1, p, cand(2, mem=12, share=1))
    ok_shared, why, _ = evaluate(2, 1, 1, p, cand(2, mem=12, share=2))
    ok_shared4, why4, _ = evaluate(4, 1, 1, p, cand(4, mem=12, share=4))
    assert ok_alone and ok_shared and not ok_shared4 and "4 ranks sharing a physical GPU" in why4


def test_legacy_profiling_still_works():
    p = Profiling(model_memory_gb=100, activation_memory_gb=10)
    plan = build_plan(2, p, cand(2), "x")
    assert (plan.dp, plan.tp, plan.pp) == (1, 2, 1)
