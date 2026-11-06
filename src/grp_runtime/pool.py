from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml


POOL_REF = "k8s:gpuresourcepools/"
POOL_GROUP, POOL_VERSION, POOL_PLURAL = "gpu.platform.io", "v1alpha1", "gpuresourcepools"
_RANK_POD = re.compile(r"-g\d+-r\d+(-gpu)?$")
_MIG_START = re.compile(r"-(\d+)$")


def is_pool_ref(path) -> bool:
    return isinstance(path, str) and path.startswith(POOL_REF)


@dataclass
class Gpu:
    id: str
    node: str
    memory_gb: float
    allocated_to: str | None = None
    physical: str = ""
    uuid: str = ""
    resource: str = ""
    model: str = ""
    mig_profile: str = ""


@dataclass
class Candidate:
    gpu_ids: list[str]
    nodes: list[str]
    interconnect: str
    gpus_per_node: int
    memory_gb: float
    rank: int = 0
    reason: str = ""
    ranks_per_physical: int = 1


@dataclass
class CandidateSet:
    snapshot_version: str
    candidates: list[Candidate] = field(default_factory=list)


class PoolClient(Protocol):
    def get_candidates(self, workload_key: str, current_gpu_ids: list[str],
                       target_gpu_count: int) -> CandidateSet: ...

    def release(self, workload_key: str, gpu_ids: list[str]) -> None: ...

    def assign(self, workload_key: str, gpu_ids: list[str]) -> None: ...


class MockPoolClient:
    source = "mock"
    atomic = False
    valid_until: str | None = None

    def __init__(self, snapshot_path: str | Path):
        data = yaml.safe_load(Path(snapshot_path).read_text(encoding="utf-8"))
        self.excluded: list[dict] = []
        self.version = data.get("snapshotVersion", "mock")
        self.interconnect = {n["name"]: n.get("interconnect", "PCIe") for n in data["nodes"]}
        self.gpus: dict[str, Gpu] = {}
        for n in data["nodes"]:
            for g in n["gpus"]:
                gid = f"{n['name']}:{g['index']}"
                self.gpus[gid] = Gpu(gid, n["name"], float(g.get("memoryGb", 0)), g.get("allocatedTo"),
                                     g.get("physical", gid), model=g.get("model", ""),
                                     mig_profile=g.get("migProfile", ""))

    def stale(self, at=None) -> bool:
        return False

    def slot_uuids(self) -> dict[tuple, str]:
        return {}

    def free_gpus(self) -> list[Gpu]:
        return [g for g in self.gpus.values() if g.allocated_to is None]

    def gpus_of(self, workload_key: str) -> list[Gpu]:
        return [g for g in self.gpus.values() if g.allocated_to == workload_key]

    def get_candidates(self, workload_key, current_gpu_ids, target_gpu_count) -> CandidateSet:
        current = [self.gpus[i] for i in current_gpu_ids]
        cands: list[Candidate] = []
        if target_gpu_count <= len(current):
            by_node = _group(current)
            for node, gs in sorted(by_node.items(), key=lambda kv: -len(kv[1])):
                if len(gs) >= target_gpu_count:
                    keep = gs[:target_gpu_count]
                    cands.append(self._cand(keep, "keep current GPUs, shrink within a single node"))
            if not cands:
                keep = current[:target_gpu_count]
                cands.append(self._cand(keep, "keep current GPUs, spread across nodes"))
        else:
            need = target_gpu_count - len(current)
            cur_nodes = {g.node for g in current}
            free = sorted(self.free_gpus(), key=lambda g: (g.node not in cur_nodes, g.node, g.id))
            if len(free) >= need:
                add = free[:need]
                same = all(g.node in cur_nodes for g in add) or not current
                cands.append(self._cand(current + add,
                                        "keep current GPUs + add free GPUs on the same node" if same
                                        else "keep current GPUs + add free GPUs on other nodes"))
        for i, c in enumerate(cands, 1):
            c.rank = i
        return CandidateSet(self.version, cands)

    def _cand(self, gs: list[Gpu], reason: str) -> Candidate:
        by_node = _group(gs)
        nodes = sorted(by_node)
        ic = self.interconnect[nodes[0]] if len(nodes) == 1 else "Network"
        per_phys: dict[str, int] = {}
        for g in gs:
            per_phys[g.physical] = per_phys.get(g.physical, 0) + 1
        return Candidate(
            gpu_ids=[g.id for g in gs], nodes=nodes, interconnect=ic,
            gpus_per_node=max(len(v) for v in by_node.values()),
            memory_gb=min(g.memory_gb for g in gs), reason=reason,
            ranks_per_physical=max(per_phys.values()),
        )

    def release(self, workload_key, gpu_ids):
        for i in gpu_ids:
            if self.gpus[i].allocated_to == workload_key:
                self.gpus[i].allocated_to = None

    def assign(self, workload_key, gpu_ids):
        for i in gpu_ids:
            self.gpus[i].allocated_to = workload_key


class GrpPoolStatusClient(MockPoolClient):
    source = "GpuResourcePool"

    def __init__(self, snapshot_path: str | Path, doc: dict | None = None):
        import hashlib
        import json
        doc = doc or yaml.safe_load(Path(snapshot_path).read_text(encoding="utf-8"))
        st = doc.get("status") or {}
        sv = st.get("sourceVersions") or {}
        digest = hashlib.sha256(json.dumps(sv, sort_keys=True).encode()).hexdigest()[:12]
        self.version = f"{doc.get('metadata', {}).get('name', 'pool')}@{digest}"
        self.source_versions = sv
        self.atomic = bool(st.get("atomicSnapshot", False))
        self.valid_until = st.get("validUntil")
        self.observed_at = st.get("observedAt")
        self.phase = st.get("phase")
        self.resources = list(st.get("resources") or [])
        self.interconnect, self.gpus, self.excluded = {}, {}, []
        counter: dict[str, int] = {}
        kinds: dict[str, set] = {}
        for r in self.resources:
            node = r.get("node", "")
            label = f"{node}/{r.get('resource', '')}"
            holder = _holder(r)
            why = []
            if r.get("nodeFlags"):
                why.append(f"nodeFlags={r['nodeFlags']}")
            if not r.get("unassigned") and not holder:
                why.append("no unassigned unit")
            if why:
                self.excluded.append({"device": label, "model": r.get("model"), "reason": ", ".join(why)})
                continue
            idx = counter.get(node, 0)
            counter[node] = idx + 1
            gid = f"{node}:{idx}"
            kinds.setdefault(node, set()).add(r.get("deviceType"))
            mig = r.get("deviceType") == "mig"
            physical = (r.get("uuid") if mig else r.get("parentUuid") or r.get("uuid")) or gid
            self.gpus[gid] = Gpu(gid, node, _mi_to_gb(r.get("capacity")), holder, physical,
                                 r.get("uuid") or "", r.get("resource") or "", r.get("model") or "",
                                 (r.get("migProfile") or "") if mig else "")
        for node, k in kinds.items():
            self.interconnect[node] = "MIG" if k == {"mig"} else "unknown"

    def slot_uuids(self) -> dict[tuple, str]:
        parents: dict[str, set] = {}
        for r in self.resources:
            parents.setdefault(r.get("node", ""), set()).add(r.get("parentUuid") or r.get("uuid"))
        index = {(n, p): f"GPU{i}" for n, ps in parents.items() for i, p in enumerate(sorted(ps, key=str))}
        out = {}
        for r in self.resources:
            node, mig = r.get("node", ""), r.get("deviceType") == "mig"
            start = r.get("placement")
            if mig and start is None:
                m = _MIG_START.search(r.get("resource") or "")
                start = int(m.group(1)) if m else None
            if not r.get("uuid") or (mig and start is None):
                continue
            gpu = index[(node, r.get("parentUuid") or r.get("uuid"))]
            out[(node, gpu, r.get("migProfile") if mig else "full", start if mig else 0)] = r["uuid"]
        return out

    def stale(self, at=None) -> bool:
        from datetime import datetime, timezone
        if not self.valid_until:
            return False
        until = datetime.fromisoformat(self.valid_until.replace("Z", "+00:00"))
        return (at or datetime.now(timezone.utc)) > until


def _holder(resource: dict) -> str | None:
    owners = resource.get("owner") or []
    if not owners or resource.get("unassigned"):
        return None
    keys = {_RANK_POD.sub("", o) for o in owners}
    return keys.pop() if len(keys) == 1 else ",".join(sorted(keys))


def _mi_to_gb(capacity: str | None) -> float:
    if not capacity:
        return 0.0
    units = {"Ki": 1 / 1024 ** 2, "Mi": 1 / 1024, "Gi": 1.0, "Ti": 1024.0}
    for suf, mul in units.items():
        if capacity.endswith(suf):
            return round(float(capacity[:-len(suf)]) * mul, 2)
    return round(float(capacity) / 1024 ** 3, 2)


def fetch_pool(name: str) -> dict:
    from kubernetes import client, config
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CustomObjectsApi().get_cluster_custom_object(POOL_GROUP, POOL_VERSION, POOL_PLURAL, name)


def load_pool(path: str | Path) -> MockPoolClient:
    if is_pool_ref(path):
        return GrpPoolStatusClient(path, fetch_pool(path[len(POOL_REF):]))
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and doc.get("kind") == "GpuResourcePool":
        return GrpPoolStatusClient(path, doc)
    return MockPoolClient(path)


def _group(gs: list[Gpu]) -> dict[str, list[Gpu]]:
    out: dict[str, list[Gpu]] = {}
    for g in gs:
        out.setdefault(g.node, []).append(g)
    return out
