#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${TAG:-0.1.0}
REGISTRY=${REGISTRY:-}
TARGETS=${TARGETS:-"operator train"}
for t in $TARGETS; do
  img="${REGISTRY}grp-runtime-$t:$TAG"
  docker build -f "deploy/image/Dockerfile.$t" -t "$img" .
  docker image inspect "$img" --format "$img {{.Id}} {{.Size}}"
  if [ "${IMPORT_K3S:-0}" = 1 ]; then
    docker save "$img" | k3s ctr images import -
  fi
  if [ "${PUSH:-0}" = 1 ]; then
    docker push "$img"
  fi
done
