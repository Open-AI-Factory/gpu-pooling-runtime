#!/usr/bin/env bash
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
K() { kubectl "$@"; }
command -v kubectl >/dev/null || K() { k3s kubectl "$@"; }

helm uninstall nvidia-dra-driver-gpu -n nvidia-dra-driver-gpu || true
K delete ns nvidia-dra-driver-gpu --ignore-not-found
K -n nvidia-device-plugin patch ds nvdp-nvidia-device-plugin --type json \
  -p '[{"op":"remove","path":"/spec/template/spec/nodeSelector/grp.keti.re.kr~1device-plugin"}]'
K -n nvidia-device-plugin rollout status ds nvdp-nvidia-device-plugin --timeout 120s
K get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
