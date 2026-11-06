#!/usr/bin/env bash
set -euo pipefail
NAMESPACE=${NAMESPACE:-grp-runtime}
KUBECTL=${KUBECTL:-kubectl}
K() { $KUBECTL "$@"; }
owner=$(K get ns "$NAMESPACE" -o jsonpath='{.metadata.labels.app\.kubernetes\.io/part-of}' 2>/dev/null || true)
if [ "$owner" != grp-runtime ]; then
  echo "abort: namespace $NAMESPACE is not labelled app.kubernetes.io/part-of=grp-runtime"; exit 1
fi
for kind in planrequests executionplans runtimeactions; do
  K -n "$NAMESPACE" delete "$kind.grp.keti.re.kr" --all --ignore-not-found --wait=true
done
K -n "$NAMESPACE" delete deploy -l app.kubernetes.io/part-of=grp-runtime --ignore-not-found
K delete ns "$NAMESPACE" --ignore-not-found --wait=true
for r in grp-operator-read grp-pool-observer grp-viewer; do
  K delete clusterrolebinding "$r-$NAMESPACE" --ignore-not-found
  K delete clusterrole "$r-$NAMESPACE" --ignore-not-found
done
K get pv "grp-checkpoints-$NAMESPACE" >/dev/null 2>&1 && \
  echo "kept PersistentVolume grp-checkpoints-$NAMESPACE (Retain); delete it manually when the data is no longer needed"
echo "CRDs grp.keti.re.kr are left installed"
