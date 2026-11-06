#!/usr/bin/env bash
set -euo pipefail
SERVER=${SERVER:?set SERVER to https://<server>:6443}
VERSION=${VERSION:-v1.36.5+k3s1}
NODE_NAME=${NODE_NAME:?set NODE_NAME}
IMAGE_TAR=${IMAGE_TAR:-}
[ "$(id -u)" = 0 ] || { echo "run with sudo -E"; exit 1; }
if [ -z "${K3S_TOKEN:-}" ]; then
  read -rsp "k3s join token (on the server: sudo cat /var/lib/rancher/k3s/server/node-token): " K3S_TOKEN; echo
fi

echo "== precheck"
hostname
nvidia-smi -L
nvidia-ctk --version | head -1

echo "== install k3s agent $VERSION as $NODE_NAME -> $SERVER"
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$VERSION" K3S_URL="$SERVER" K3S_TOKEN="$K3S_TOKEN" \
  sh -s - agent --node-name "$NODE_NAME" --node-label nvidia.com/gpu.present=true
for _ in $(seq 1 30); do systemctl is-active -q k3s-agent && break; sleep 2; done
systemctl is-active k3s-agent

echo "== nvidia runtime in containerd"
for _ in $(seq 1 30); do grep -q nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml 2>/dev/null && break; sleep 2; done
grep -c '"nvidia"' /var/lib/rancher/k3s/agent/etc/containerd/config.toml

if [ -f "$IMAGE_TAR" ]; then
  echo "== import $IMAGE_TAR"
  k3s ctr images import "$IMAGE_TAR"
fi
echo "done: check on the server with kubectl get nodes"
