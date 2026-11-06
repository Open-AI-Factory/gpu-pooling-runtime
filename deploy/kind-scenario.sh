#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=/root/venvs/grp
NS=tenant-a
EXAMPLES=("${@:-ra-scaledown-mock ra-reconfigure-fail-mock ra-preparation-timeout-mock}")

kind get clusters | grep -qx grp || kind create cluster --name grp
kubectl wait --for=condition=Ready node --all --timeout=120s >/dev/null
kubectl apply -f deploy/crd-runtimeaction.yaml >/dev/null
kubectl create ns $NS --dry-run=client -o yaml | kubectl apply -f - >/dev/null

pkill -f "^$VENV/bin/python3? $VENV/bin/kopf" ; sleep 1
nohup $VENV/bin/kopf run -m grp_runtime.runtime_coordination.runtime_action_controller.operator --all-namespaces > /root/operator.log 2>&1 &
sleep 5

kubectl delete ra --all -n $NS --ignore-not-found >/dev/null
for f in ${EXAMPLES[@]}; do kubectl apply -n $NS -f scenarios/runtime-action/$f.yaml; done
echo "---- kubectl get ra -w"
timeout 12 kubectl get ra -n $NS -w
echo "---- final"
kubectl get ra -n $NS
