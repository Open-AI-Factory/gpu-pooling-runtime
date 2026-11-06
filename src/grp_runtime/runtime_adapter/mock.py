from __future__ import annotations

import time
from pathlib import Path

import yaml

from ..models import ExecutionPlan, ParallelConfig, RuntimeActionSpec, WorkloadState
from ..pool import MockPoolClient
from .base import AdapterError, RuntimeAdapter


class MockAdapter(RuntimeAdapter):
    def __init__(self, name: str, pool: MockPoolClient, snapshot_path: str | Path):
        self.name = name
        self.pool = pool
        data = yaml.safe_load(Path(snapshot_path).read_text(encoding="utf-8"))
        self.parallel = {
            w["ref"]: ParallelConfig(**w["parallel"]) for w in data.get("workloads", [])
        }
        self._sp_requested_at: float | None = None

    def current_state(self, spec):
        key = spec.workload_ref.key
        gpus = [g.id for g in self.pool.gpus_of(key)]
        if not gpus or key not in self.parallel:
            raise AdapterError(f"workload {key} is not running (not in mock snapshot)")
        return WorkloadState(gpus, self.parallel[key], {"runtime": spec.runtime, "mode": "mock"})

    def preparation_ready(self, spec, plan):
        if spec.adapter_options.get("failAt") == "Preparing":
            return False, "mock: Agent not ready (injected failure)"
        return True, "preparation=Ready"

    def request_safe_point(self, spec):
        self._sp_requested_at = time.monotonic()

    def safe_point_ready(self, spec):
        if spec.adapter_options.get("failAt") == "SafePoint":
            return False
        wait = float(spec.adapter_options.get("safePointAfterSeconds", 1.0))
        return self._sp_requested_at is not None and time.monotonic() - self._sp_requested_at >= wait

    def cancel_safe_point(self, spec):
        self._sp_requested_at = None

    def reconfigure(self, spec, plan: ExecutionPlan):
        time.sleep(0.3)
        if spec.adapter_options.get("failAt") == "Reconfigure":
            raise AdapterError("mock: rank restart failed during reconfigure (injected failure)")
        self.parallel[spec.workload_ref.key] = plan.parallel
        return WorkloadState(plan.gpu_ids, plan.parallel, {"runtime": spec.runtime, "mode": "mock"})

    def rollback(self, spec, before):
        self.parallel[spec.workload_ref.key] = before.parallel
