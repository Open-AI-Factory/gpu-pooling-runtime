#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAMESPACE=${NAMESPACE:-grp-runtime}
REGISTRY=${REGISTRY:-}
TAG=${TAG:-0.1.0}
TRAIN_TAG=${TRAIN_TAG:-0.1.1}
STORAGE_CLASS=${STORAGE_CLASS:?set STORAGE_CLASS, e.g. ai-factory-cephfs, or a manual class name together with DEV_HOSTPATH}
CKPT_SIZE=${CKPT_SIZE:-20Gi}
DEV_HOSTPATH=${DEV_HOSTPATH:-}
WITH_CRDS=${WITH_CRDS:-0}
WITH_OBSERVER=${WITH_OBSERVER:-0}
WITH_VIEWER=${WITH_VIEWER:-0}
WITH_ARGO=${WITH_ARGO:-0}
QUOTA=${QUOTA:-0}
QUOTA_GPUS=${QUOTA_GPUS:-2}
QUOTA_PODS=${QUOTA_PODS:-20}
MODE=${MODE:-apply}
KUBECTL=${KUBECTL:-kubectl}
K() { $KUBECTL "$@"; }
OUT=${OUT:-$(mktemp -d)}

render() {
  sed -e "s#__NAMESPACE__#$NAMESPACE#g" -e "s#__REGISTRY__#$REGISTRY#g" -e "s#__TAG__#$TAG#g" \
      -e "s#__STORAGE_CLASS__#$STORAGE_CLASS#g" -e "s#__CKPT_SIZE__#$CKPT_SIZE#g" \
      -e "s#__DEV_HOSTPATH__#$DEV_HOSTPATH#g" -e "s#__QUOTA_GPUS__#$QUOTA_GPUS#g" \
      -e "s#__QUOTA_PODS__#$QUOTA_PODS#g" -e "s#__TRAIN_TAG__#$TRAIN_TAG#g" "$HERE/$1" > "$OUT/$1"
  echo "$OUT/$1"
}

FILES=$(render namespace.yaml; render rbac.yaml)
[ "$QUOTA" = 1 ] && FILES="$FILES $(render quota.yaml)"
[ -n "$DEV_HOSTPATH" ] && FILES="$FILES $(render checkpoints-dev-hostpath.yaml)"
FILES="$FILES $(render checkpoints-pvc.yaml) $(render operator.yaml)"
[ "$WITH_OBSERVER" = 1 ] && FILES="$FILES $(render observer-dev.yaml)"
[ "$WITH_VIEWER" = 1 ] && FILES="$FILES $(render viewer.yaml)"
[ "$WITH_ARGO" = 1 ] && FILES="$FILES $(render argo-templates.yaml)"
echo "== rendered to $OUT"
[ "$MODE" = render ] && exit 0

echo "== precheck"
owner=$(K get ns "$NAMESPACE" -o jsonpath='{.metadata.labels.app\.kubernetes\.io/part-of}' 2>/dev/null || true)
if K get ns "$NAMESPACE" >/dev/null 2>&1 && [ "$owner" != grp-runtime ]; then
  echo "abort: namespace $NAMESPACE exists and is not labelled app.kubernetes.io/part-of=grp-runtime"; exit 1
fi
missing=""
for crd in planrequests executionplans runtimeactions; do
  K get crd "$crd.grp.keti.re.kr" >/dev/null 2>&1 || missing="$missing $crd.grp.keti.re.kr"
done
if [ -n "$missing" ] && [ "$WITH_CRDS" != 1 ]; then
  echo "abort: CRD(s) missing:$missing (rerun with WITH_CRDS=1 after approval)"; exit 1
fi
if [ -z "$DEV_HOSTPATH" ] && ! K get storageclass "$STORAGE_CLASS" >/dev/null 2>&1; then
  echo "abort: StorageClass $STORAGE_CLASS not found"; exit 1
fi
if [ "$WITH_ARGO" = 1 ] && ! K get crd workflowtemplates.argoproj.io >/dev/null 2>&1; then
  echo "abort: Argo Workflows CRDs not found (WITH_ARGO=1)"; exit 1
fi
K get gpuresourcepools.gpu.platform.io gpu-pool >/dev/null 2>&1 || echo "warning: GpuResourcePool gpu-pool not found"

CRDS=""
[ "$WITH_CRDS" = 1 ] && CRDS="$HERE/../crd-planrequest.yaml $HERE/../crd-executionplan.yaml $HERE/../crd-runtimeaction.yaml"
if [ "$MODE" = dry-run ]; then
  for f in $CRDS $FILES; do K apply --dry-run=server -f "$f"; done
  exit 0
fi
if [ -n "$DEV_HOSTPATH" ] && [ "$(K get pv "grp-checkpoints-$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null)" = Released ]; then
  K patch pv "grp-checkpoints-$NAMESPACE" --type json \
    -p '[{"op":"remove","path":"/spec/claimRef/uid"},{"op":"remove","path":"/spec/claimRef/resourceVersion"}]'
fi
for f in $CRDS $FILES; do K apply -f "$f"; done
K -n "$NAMESPACE" rollout status deploy/grp-operator --timeout=180s
[ "$WITH_OBSERVER" = 1 ] && K -n "$NAMESPACE" rollout status deploy/grp-pool-observer --timeout=180s
[ "$WITH_VIEWER" = 1 ] && K -n "$NAMESPACE" rollout status deploy/grp-viewer --timeout=180s
K -n "$NAMESPACE" get deploy,pvc,sa
