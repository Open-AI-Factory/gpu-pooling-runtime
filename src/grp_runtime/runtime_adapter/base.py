from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import ExecutionPlan, RuntimeActionSpec, WorkloadState


class AdapterError(Exception):
    pass


class RuntimeAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def current_state(self, spec: RuntimeActionSpec) -> WorkloadState:
        pass

    def prepare(self, spec: RuntimeActionSpec, plan: ExecutionPlan) -> None:
        pass

    def preparation_ready(self, spec: RuntimeActionSpec, plan: ExecutionPlan) -> tuple[bool, str]:
        return True, "preparation=Ready"

    def verify_checkpoint(self, spec: RuntimeActionSpec) -> tuple[bool, str]:
        return True, "serviceConfig=Valid"

    def verify_ready(self, spec: RuntimeActionSpec) -> tuple[bool, str]:
        return True, "ready=true response=OK"

    @abstractmethod
    def request_safe_point(self, spec: RuntimeActionSpec) -> None:
        pass

    @abstractmethod
    def safe_point_ready(self, spec: RuntimeActionSpec) -> bool: ...

    @abstractmethod
    def cancel_safe_point(self, spec: RuntimeActionSpec) -> None:
        pass

    @abstractmethod
    def reconfigure(self, spec: RuntimeActionSpec, plan: ExecutionPlan) -> WorkloadState:
        pass

    @abstractmethod
    def rollback(self, spec: RuntimeActionSpec, before: WorkloadState) -> None:
        pass

    def plan_settings(self, spec: RuntimeActionSpec, plan: ExecutionPlan) -> tuple[dict | None, dict | None, str]:
        return None, None, f"adapter {self.name} cannot apply execution settings"

    def observe(self, spec: RuntimeActionSpec) -> dict:
        return {}

    def finalize(self, spec: RuntimeActionSpec) -> None:
        pass
