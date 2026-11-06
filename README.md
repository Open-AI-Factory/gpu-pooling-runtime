# GRP Runtime Engine Controller

A runtime engine control prototype that changes the GPU configuration of a training workload while it is running.
It takes an execution request through **ExecutionPlan generation → placement → execution**, and when a RuntimeAction arrives it handles
**Safe Point → checkpoint → restart with the new configuration → resume**, recording the result in the status.

| Component | Code |
|---|---|
| Execution Plan Controller | `execution_plan_controller/` — `intake.py` (request validation → requirement analysis → DP/TP/PP decision → ExecutionPlan creation and registration), `planner.py`, `placement.py`, `execution_plan.py` |
| Runtime Action Controller | `runtime_coordination/runtime_action_controller/` — `controller.py` (state transitions), `operator.py` (kopf watch · sequential processing · duplicate suppression) |
| Adapter Manager · Registry | `runtime_coordination/adapter_manager/registry.py` (capability declaration, adapter selection) |
| Runtime Adapter | `runtime_adapter/` mock · local (single process) · torchrun (multi-process) · k8s (rank Pods, `workload.py`) |
| Runtime Agent | `agent/hook.py` (hook added to training code), `agent/channel.py` (file · ConfigMap channel) |
| Resource pool integration | `pool.py` (candidate · snapshot interface + mock) |
| Service recovery | `recovery/` (partial reconfiguration based on a component dependency graph) |
| CLI | `cli.py` (`run` / `plan`), `keticti.py` (`runtime agents/watch`, `plan create/request`, `quota set`) |

## Layout

| Directory | Description |
|---|---|
| `src/grp_runtime/` | Source code |
| `scenarios/runtime-action/` | RuntimeAction scenarios (scale down · scale up · deadline exceeded · reconfiguration failure · no feasible plan) |
| `scenarios/execution-plan/` | ExecutionPlan examples |
| `scenarios/plan-request/` | Execution request (PlanRequest) examples — normal · pinned GPUs · invalid input · insufficient resources · version mismatch |
| `scenarios/recovery/` | Service definitions · recovery requests |
| `workloads/` | DDP training workload with the hook applied, and its profiling values |
| `experiments/` | Training result comparison before/after reconfiguration |
| `mock/` | Mock resource pool snapshots |
| `deploy/` | CRDs · RBAC, WSL · Kind setup, operator launcher, scenario scripts |
| `tests/` | Automated tests |

## State transitions

```
Accepted     request validated
Preparing    preparation=Ready       execution plan created, agent connected and isolation values matched (ReadyToRun)
Checkpoint   serviceConfig=Valid     Safe Point reached, checkpoint saved and verified (size, digest)
Reconfigure  restart=Running         restarted with the new configuration (mode=restart)
Succeeded    ready=true response=OK  agent reconnected, isolation values matched, training progressing after restart

Aborted      reason=… previous=…
  failure before change → previous=Kept     (InvalidSpec / InvalidAction / WorkloadNotFound / InsufficientResources /
                                             NoFeasiblePlan / AdapterRejected / PreparationTimeout / DuplicateRequest)
  failure after change  → previous=Restored (ReconfigureFailed, previous configuration restored) / Lost (restore failed)
```

`safePolicy.timeoutSeconds` is the preparation deadline before the change (Preparing + Checkpoint). When it is exceeded, the action ends with `PreparationTimeout` and the existing configuration is kept.

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,k8s]" numpy
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # for GPU runs
```

## Local runs

```bash
python -m grp_runtime plan scenarios/runtime-action/ra-scaledown-mock.yaml        # candidate combinations and selection reason
python -m grp_runtime run  scenarios/runtime-action/ra-scaledown-mock.yaml        # run the state transitions
python -m grp_runtime run  scenarios/runtime-action/ra-scaledown-local-gpu.yaml   # real GPU
python -m grp_runtime run  scenarios/runtime-action/ra-scaledown-torchrun.yaml    # torchrun DDP (Linux/WSL)
python -m pytest -q
```

Results are written to `runs/<name>/status.yaml` (final status) and `runs/<name>/events.jsonl` (per-phase records).

## Runtime Hook

Adding a few lines to the training code enables Safe Point · checkpoint · resume (see `workloads/train_ddp.py`).

```python
from grp_runtime.agent import RuntimeHook

hook = RuntimeHook()
start = hook.restore(model, optimizer)
for step in range(start, max_steps):
    ...
    if hook.step_end(step + 1, model, optimizer, loss=loss.item()):
        break
```

- Rank 0 reads the request and shares it with `all_reduce`, so all ranks stop at the same step. Rank 0 saves, then all ranks exit after a `barrier`
- Reconfiguration relaunches torchrun with the new world size and resumes from the checkpoint
- Checkpoints are stored in a folder per RuntimeAction, and only the latest `GRP_CKPT_KEEP` (default 3) are kept per folder

## Planner memory check

```
memory per rank = (weights + gradients + optimizer) ÷ (TP×PP) + (global batch ÷ DP) × activation per sample ÷ PP
check: memory per rank × ranks sharing the same physical GPU ≤ physical GPU memory × 0.9
```

Among the combinations that fit in memory, the one with the smallest model split (TP×PP) is chosen. When a split is needed, TP is preferred on a single node with a fast interconnect (NVLink), and PP otherwise.

## Execution settings

`launch.settings` fixes the effective batch and lets the plan choose micro-batch and gradient accumulation for the selected parallelism. The result is passed to ranks in `AI_FACTORY_RUNTIME_CONFIG` (`runtime-plan-env-v1`) and recomputed on every reconfiguration.

```yaml
settings:
  effectiveBatch: 1024
  maxMicroBatch: 256
  profileConfigMap: ddp-profiles
  identity: {precision: float32}
  allowSynthetic: false
```

- Profiles use the EP3 `runtime_plan` profile format. They come from `profiles` (inline list or bundle) or from `profiles.json` in the ConfigMap named by `profileConfigMap` (a list, or an EP3 bundle with `acceptedProfiles`).
- A profile matches when the effective batch, DP/TP/PP, node count, GPUs per node, GPU model, mode and MIG profile all equal the allocated devices, and the optional `identity` fields (`model`, `precision`, `signature`) agree.
- Among matching profiles, those with a changed effective batch, a micro-batch over `maxMicroBatch`, failed quality, a memory budget below the measured peak or above device memory, or synthetic evidence (unless `allowSynthetic`) are dropped. The one with the highest measured throughput is chosen (`source: profile`).
- No matching profile: the rule is used (largest divisor of the per-rank batch up to `maxMicroBatch`, `source: rule`) and `profiles.status: needsProfiling` is recorded.
- Matching profiles but none usable: the plan is rejected with `noFeasiblePlan`; during reconfiguration this rolls back to the previous configuration.
- `settings.batch: {microBatch, gradientAccumulation}` pins the batch (`source: pinned`); it is used for measurement runs and is rejected when it does not give the effective batch for the selected DP.
- Example: `scenarios/profiles/ddp-profiles-synthetic.yaml` with `scenarios/plan-request/pr-train-ddp4-profiles-k3s-mig.yaml` and `scenarios/runtime-action/ra-ddp4p-to-dp2-k3s-mig.yaml`.

### Applying settings to a running workload

Two RuntimeAction actions keep the parallelism and GPUs of the workload unchanged.

- `Observe`: records the current state in `status.observation` (step, world size, readiness, planned settings with profile, applied settings, `settingsMatch`). Nothing is restarted; `target` must be empty.
- `ApplySettings`: merges `target.settings` into the plan's `launch.settings` (for example a new `profileConfigMap` or `maxMicroBatch`) and selects settings again for the current layout before touching the workload.
  - Not usable (`noFeasiblePlan`, invalid settings): `Aborted reason=SettingsRejected previous=Kept`, no restart.
  - Same batch as now: `Aborted reason=InvalidAction`, no restart.
  - Otherwise Safe Point, checkpoint, restart at the same layout with the new settings, and `settings=Applied` check. On failure the previous `launch.settings` and planning are restored (`previous=Restored`).

```yaml
spec:
  executionPlanRef: ddp4s-plan
  action: ApplySettings
  target:
    settings: {profileConfigMap: train-ddp-profiles}
```

Scenarios: `ra-ddp4s-observe.yaml`, `ra-ddp4s-apply-profiles.yaml`, `ra-ddp4s-apply-infeasible.yaml`, `ra-ddp4s-apply-fail-rollback.yaml` in `scenarios/runtime-action/`.

These map the EP3 `runtime.aifactory.dev` RuntimeAction operations: `ApplyPlan` → `ApplySettings` (delivered by checkpoint restart and environment instead of a sidecar), `Observe` → `Observe`. The EP3 CRD and controller are not used or changed.

### Measuring profiles

`deploy/profiles/measure_profiles.py` runs one PlanRequest per `DPxMicroBatch` combination through the normal launch path with a pinned batch, reads the throughput and peak reserved memory that `train_ddp.py --measure-warmup` reports at the end, and writes an EP3-style bundle (`acceptedProfiles`, `excludedProfiles`, `evidenceManifest`) and optionally a ConfigMap.

```bash
KUBECONFIG=/etc/rancher/k3s/k3s.yaml PYTHONPATH=src python deploy/profiles/measure_profiles.py \
  --template scenarios/plan-request/pr-train-ddp4-settings-k3s-mig.yaml --gpus <uuid>,<uuid>,... \
  --configs 4x64,4x128,2x128,1x256 --effective-batch 1024 --steps 200 --warmup 20 \
  --model-id grp/train_ddp-mlp --out runs/profiles/train-ddp.json --configmap train-ddp-profiles
```

- Hardware (model, MIG profile, device memory) comes from the GpuResourcePool entries of the given devices; mixed hardware is refused.
- Memory budget is the measured peak × `--headroom` (default 1.2), capped at device memory. Quality passes when the loss is finite and decreased.
- Runs wait until the pool no longer lists the previous run's pods before starting the next one.
- `scenarios/profiles/train-ddp-mig-measured.yaml` holds 11 profiles measured on RTX PRO 6000 MIG 1g.24gb (gloo, effective batch 1024).

## Kubernetes (multi-node Kind)

```bash
bash deploy/setup-wsl.sh                  # install Docker Engine · kubectl · kind (WSL Ubuntu)
bash deploy/kind-up.sh                    # build image · create cluster · register CRDs/RBAC
bash deploy/operator.sh start             # start the operator (logs: bash deploy/operator.sh log)

kubectl apply -f scenarios/plan-request/pr-train-kind.yaml   # execution request → ExecutionPlan → placement → rank Pods
kubectl get planrequest
kubectl get executionplan

keticti runtime agents train              # agent connection · isolation value comparison
keticti runtime watch                     # RuntimeAction progress
keticti plan request scenarios/plan-request/pr-train-kind.yaml   # generate a plan only, no cluster needed
```

`deploy/slide15-scenario.sh` reproduces the whole flow in `setup` / `resize` / `timeout` / `restore` / `dup` / `request` / `reset` steps,
and `deploy/capture-scenario.sh all` saves the output of each step to `runs/capture/<timestamp>/`.

## Service recovery

```
image ───────────────┐
volume ── model ─────┤
config ──────────────┼── runtime ── endpoint
gpu ─────────────────┘
```

At deployment, the configuration of each component is recorded in `record.json`. When a recovery request arrives, every component is diagnosed,
and only the faulty components and the components that depend on them are rebuilt, in dependency order. If nothing is faulty, nothing is rebuilt (`NO_FAULT_FOUND`).

```
Received → Diagnosing → Reacquiring → Restarting → Verifying → Recovered
   ├─ same requestId → Duplicate
   └─ recovery already running for the same service → Merged
```

```bash
python -m grp_runtime svc-deploy    scenarios/recovery/svc-mlp-serve-local.yaml
python -m grp_runtime svc-fail      scenarios/recovery/svc-mlp-serve-local.yaml runtime
python -m grp_runtime recover       scenarios/recovery/rr-detector-runtime.yaml scenarios/recovery/rr-pipeline-runtime.yaml
python -m grp_runtime recover-bench scenarios/recovery/svc-mlp-serve-local.yaml
python -m grp_runtime svc-stop      scenarios/recovery/svc-mlp-serve-local.yaml
```
