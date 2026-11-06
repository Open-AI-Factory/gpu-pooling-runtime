from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kubernetes import client

from ..agent.channel import ConfigMapChannel
from ..execution_plan_controller.execution_plan import claim_manifest

LABEL = "grp.keti.re.kr/workload"
GEN = "grp.keti.re.kr/generation"
RANK = "grp.keti.re.kr/rank"
CLAIM_API, CLAIM_PLURAL = ("resource.k8s.io", "v1"), "resourceclaims"
ContainerClaim = getattr(client, "CoreV1ResourceClaim", None) or getattr(client, "V1ResourceClaim")
POD_ROOT = "/work"
HOST_ROOT = Path(__file__).resolve().parents[3]
HEARTBEAT_TIMEOUT = 10.0
STOP_GRACE = 1


class LaunchError(RuntimeError):
    pass


def api_message(e: Exception) -> str:
    try:
        return json.loads(e.body).get("message") or str(e)
    except (AttributeError, TypeError, ValueError):
        return str(e)


def cm_name(workload: str) -> str:
    return f"grp-{workload}"


def svc_name(workload: str) -> str:
    return f"{workload}-ranks"


def to_host_path(pod_path: str) -> Path:
    return HOST_ROOT / Path(pod_path).relative_to(POD_ROOT)


CKPT_ROOT = f"{POD_ROOT}/runs/k8s"


def ckpt_dir(ns: str, workload: str, tag: str) -> str:
    return f"{CKPT_ROOT}/{ns}_{workload}/ckpt/{tag}"


def _mem_mb(q: str) -> int:
    units = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024}
    for u, f in units.items():
        if q.endswith(u):
            return int(float(q[:-len(u)]) * f)
    return int(int(q) / 2**20)


class Workload:
    def __init__(self, ns: str, name: str, core: client.CoreV1Api | None = None,
                 custom: client.CustomObjectsApi | None = None):
        self.ns, self.name = ns, name
        self.core = core or client.CoreV1Api()
        self._custom = custom
        self.channel = ConfigMapChannel(cm_name(name), ns, api=self.core)

    @property
    def custom(self) -> client.CustomObjectsApi:
        if self._custom is None:
            self._custom = client.CustomObjectsApi()
        return self._custom

    def claims(self, generation: int | None = None) -> list[dict]:
        sel = f"{LABEL}={self.name}" + (f",{GEN}={generation}" if generation is not None else "")
        return self.custom.list_namespaced_custom_object(*CLAIM_API, self.ns, CLAIM_PLURAL,
                                                         label_selector=sel).get("items", [])

    def claim_bindings(self, generation: int) -> list[dict]:
        out = []
        for c in sorted(self.claims(generation), key=lambda c: int(c["metadata"]["labels"].get(RANK, 0))):
            md = c["metadata"]
            results = (((c.get("status") or {}).get("allocation") or {}).get("devices") or {}).get("results") or []
            out.append({"rank": int(md["labels"].get(RANK, 0)), "claim": md["name"],
                        "uuid": (md.get("annotations") or {}).get("grp.keti.re.kr/device-uuid"),
                        "allocated": bool(results), "node": results[0].get("pool") if results else None,
                        "resource": results[0].get("device") if results else None,
                        "reservedFor": [r.get("name") for r in (c.get("status") or {}).get("reservedFor") or []]})
        return out

    def pods(self) -> list:
        return self.core.list_namespaced_pod(self.ns, label_selector=f"{LABEL}={self.name}").items

    def state(self) -> dict:
        return self.channel.read("state") or {}

    def agents(self) -> list[dict]:
        data = self.channel.read_all()
        return sorted((v for k, v in data.items() if k.startswith("agent-")), key=lambda a: a["rank"])

    def ensure_base(self, owners: list | None = None) -> None:
        try:
            self.core.read_namespaced_config_map(cm_name(self.name), self.ns)
        except client.ApiException as e:
            if e.status != 404:
                raise
            self.core.create_namespaced_config_map(self.ns, client.V1ConfigMap(
                metadata=client.V1ObjectMeta(name=cm_name(self.name), labels={LABEL: self.name},
                                             owner_references=owners), data={}))
        try:
            self.core.read_namespaced_service(svc_name(self.name), self.ns)
        except client.ApiException as e:
            if e.status != 404:
                raise
            self.core.create_namespaced_service(self.ns, client.V1Service(
                metadata=client.V1ObjectMeta(name=svc_name(self.name), labels={LABEL: self.name},
                                             owner_references=owners),
                spec=client.V1ServiceSpec(cluster_ip="None", publish_not_ready_addresses=True,
                                          selector={LABEL: self.name},
                                          ports=[client.V1ServicePort(name="dist", port=29500)])))

    def reset_channel(self) -> None:
        data = self.channel.read_all()
        if data:
            self.core.patch_namespaced_config_map(cm_name(self.name), self.ns,
                                                  {"data": {k: None for k in data}})

    def launch(self, world: int, nodes: list[str], runtime: str, launch: dict[str, Any],
               generation: int, resume: str | None, ckpt_tag: str,
               parallelism: dict[str, int] | None = None, owner: dict | None = None,
               plan: dict[str, Any] | None = None) -> list[dict]:
        try:
            return self._launch(world, nodes, runtime, launch, generation, resume, ckpt_tag, parallelism, owner, plan)
        except client.ApiException as e:
            self.teardown()
            raise LaunchError(f"launch of {self.name} generation {generation} failed and was cleaned up: "
                              f"{api_message(e)}") from e

    def _launch(self, world: int, nodes: list[str], runtime: str, launch: dict[str, Any],
                generation: int, resume: str | None, ckpt_tag: str,
                parallelism: dict[str, int] | None = None, owner: dict | None = None,
                plan: dict[str, Any] | None = None) -> list[dict]:
        owners = [client.V1OwnerReference(api_version=owner["apiVersion"], kind=owner["kind"],
                                          name=owner["name"], uid=owner["uid"])] if owner else None
        self.ensure_base(owners)
        self.reset_channel()
        cpu, mem = str(launch.get("cpu", "1")), str(launch.get("memory", "1Gi"))
        gpus = int(launch.get("gpu", 0))
        python = launch.get("python", "python")
        host_root = None if launch.get("codeInImage") else str(launch.get("hostRoot", POD_ROOT))
        host_mounts = list(launch.get("hostMounts", []))
        ckpt_claim = (launch.get("checkpoint") or {}).get("claimName")
        mounts = ([client.V1VolumeMount(name="work", mount_path=POD_ROOT)] if host_root else []) +             ([client.V1VolumeMount(name="checkpoints", mount_path=CKPT_ROOT)] if ckpt_claim else []) + [
            client.V1VolumeMount(name=f"host{i}", mount_path=p, read_only=True) for i, p in enumerate(host_mounts)]
        volumes = ([client.V1Volume(name="work", host_path=client.V1HostPathVolumeSource(path=host_root))]
                   if host_root else []) +             ([client.V1Volume(name="checkpoints", persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=ckpt_claim))] if ckpt_claim else []) + [
            client.V1Volume(name=f"host{i}", host_path=client.V1HostPathVolumeSource(path=p))
            for i, p in enumerate(host_mounts)]
        plan = plan or {}
        dra = gpus > 0
        claims = {c["rank"]: c for c in (plan.get("deviceBinding") or {}).get("claims") or []}
        if dra and sorted(claims) != list(range(world)):
            raise ValueError(f"GPU ranks need a DRA device claim for each of {world} ranks in the plan "
                             f"(plan has ranks {sorted(claims)})")
        limits = {"cpu": cpu, "memory": mem}
        script = launch.get("script", "workloads/train_ddp.py")
        args = " ".join(map(str, launch.get("args", [])))
        strategy = runtime if runtime in ("ddp", "fsdp") else "ddp"
        comm = plan.get("communication") or {}
        by_rank = {x["rank"]: x for x in plan.get("rankMap") or []}
        groups = plan.get("processGroups")
        master = f"{self.name}-g{generation}-r{comm.get('masterRank', 0)}"
        requested = []
        for r in range(world):
            pod = f"{self.name}-g{generation}-r{r}"
            coord = by_rank.get(r)
            labels = {LABEL: self.name, GEN: str(generation), RANK: str(r)}
            claim = f"{pod}-gpu" if dra else None
            if dra:
                self.custom.create_namespaced_custom_object(*CLAIM_API, self.ns, CLAIM_PLURAL, claim_manifest(
                    claim, claims[r], labels, [owner] if owner else None))
            env = {"RANK": str(r), "WORLD_SIZE": str(world), "LOCAL_RANK": "0",
                   "MASTER_ADDR": f"{master}.{svc_name(self.name)}.{self.ns}.svc.cluster.local",
                   "MASTER_PORT": str(comm.get("port", 29500)),
                   **({"GRP_DIST_BACKEND": comm["backend"]} if comm.get("backend") else {}),
                   **({"GRP_PROCESS_GROUPS": json.dumps(groups, separators=(",", ":"))} if groups else {}),
                   **({f"GRP_{a.upper()}_RANK": str(coord[a]) for a in ("dp", "tp", "pp")} if coord else {}),
                   "GRP_CONFIGMAP": cm_name(self.name),
                   "GRP_CKPT_DIR": ckpt_dir(self.ns, self.name, ckpt_tag), "GRP_RESUME": resume or "",
                   "OMP_NUM_THREADS": cpu.rstrip("m") if not cpu.endswith("m") else "1",
                   "MALLOC_ARENA_MAX": "2",
                   **({} if gpus else {"CUDA_VISIBLE_DEVICES": ""}),
                   **{f"GRP_{k.upper()}": str(v) for k, v in (parallelism or {}).items()},
                   **({"AI_FACTORY_RUNTIME_CONFIG": json.dumps(plan["settings"], sort_keys=True)}
                      if plan.get("settings") else {}),
                   **{k: str(v) for k, v in launch.get("env", {}).items()}}
            envs = [client.V1EnvVar(name=k, value=v) for k, v in env.items()] + [
                client.V1EnvVar(name=k, value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(field_path=f)))
                for k, f in (("NODE_NAME", "spec.nodeName"), ("POD_NAME", "metadata.name"),
                             ("POD_NAMESPACE", "metadata.namespace"))]
            cmd = f"{python} {POD_ROOT}/{script} --strategy {strategy} {args}; exit $?"
            self.core.create_namespaced_pod(self.ns, client.V1Pod(
                metadata=client.V1ObjectMeta(name=pod, labels=labels, owner_references=owners),
                spec=client.V1PodSpec(
                    hostname=pod, subdomain=svc_name(self.name), restart_policy="Never",
                    service_account_name=launch.get("serviceAccount", "grp-agent"),
                    runtime_class_name=launch.get("runtimeClass"),
                    node_selector={"kubernetes.io/hostname": nodes[r]},
                    resource_claims=[client.V1PodResourceClaim(name="gpu", resource_claim_name=claim)] if dra else None,
                    containers=[client.V1Container(
                        name="trainer", image=launch.get("image", "grp-runtime-cpu:0.1"),
                        image_pull_policy="IfNotPresent", command=["sh", "-c", cmd], env=envs,
                        working_dir=POD_ROOT,
                        resources=client.V1ResourceRequirements(requests={"cpu": cpu, "memory": mem},
                                                                 limits=limits,
                                                                 claims=[ContainerClaim(name="gpu")] if dra else None),
                        volume_mounts=mounts)],
                    volumes=volumes,
                )))
            requested.append({"rank": r, "node": nodes[r], "cpuLimit": float(cpu.rstrip("m")) / (
                1000 if cpu.endswith("m") else 1), "memoryLimitMb": _mem_mb(mem),
                "device": "cuda" if gpus else "cpu", "world": world,
                **({"plannedGpu": coord["gpu"]} if coord else {}),
                **({"claim": claim, "plannedUuid": claims[r]["uuid"]} if dra else {})})
        return requested

    def outcome(self, generation: int) -> str | None:
        phases = [p.status.phase for p in self.pods() if (p.metadata.labels or {}).get(GEN) == str(generation)]
        if not phases:
            return None
        if "Failed" in phases:
            return "Failed"
        return "Succeeded" if all(ph == "Succeeded" for ph in phases) else None

    def teardown(self, timeout: float = 60) -> None:
        for p in self.pods():
            try:
                self.core.delete_namespaced_pod(p.metadata.name, self.ns, grace_period_seconds=STOP_GRACE)
            except client.ApiException as e:
                if e.status != 404:
                    raise
        end = time.monotonic() + timeout
        while self.pods() and time.monotonic() < end:
            time.sleep(0.5)
        try:
            self.custom.delete_collection_namespaced_custom_object(*CLAIM_API, self.ns, CLAIM_PLURAL,
                                                                   label_selector=f"{LABEL}={self.name}")
        except client.ApiException as e:
            if e.status != 404:
                raise
        while self.claims() and time.monotonic() < end:
            time.sleep(0.5)

    def readiness(self, requested: list[dict]) -> dict:
        agents = {a["rank"]: a for a in self.agents()}
        world = len(requested)
        nowt = time.time()
        connected = [r for r, a in agents.items() if a.get("hook") == "connected"
                     and nowt - a.get("updatedAt", 0) <= HEARTBEAT_TIMEOUT and a.get("world") == world]
        stale = [r for r, a in agents.items() if a.get("hook") == "connected"
                 and nowt - a.get("updatedAt", 0) > HEARTBEAT_TIMEOUT]
        mismatches = []
        for req in requested:
            a = agents.get(req["rank"])
            if not a:
                continue
            iso = a.get("isolation", {})
            actual = {"node": a.get("node"), "cpuLimit": iso.get("cpuLimit"),
                      "memoryLimitMb": iso.get("memoryLimitMb"), "device": iso.get("device"),
                      "world": a.get("world")}
            for k, v in actual.items():
                if req[k] != v:
                    mismatches.append(f"rank{req['rank']}.{k}: requested {req[k]} != actual {v}")
        all_connected = len(connected) == world
        if all_connected and not mismatches:
            isolation = "requested-match"
        elif mismatches:
            isolation = "mismatch"
        else:
            isolation = "pending"
        ready = all_connected and not mismatches
        hook = "connected" if all_connected else ("partial" if connected else "disconnected")
        return {"hook": hook, "ranks": sorted(connected), "world": world, "isolation": isolation,
                "state": "ReadyToRun" if ready else "NotReady", "mismatches": mismatches,
                "staleRanks": stale,
                "detail": None if ready else (f"unresponsive ranks {stale}" if stale else
                                              f"connected ranks {sorted(connected)}/{world}"
                                              + (f", mismatches {mismatches}" if mismatches else ""))}
