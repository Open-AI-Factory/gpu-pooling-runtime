#!/usr/bin/env bash
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
NODE=${NODE:-$(hostname)}
B=${BACKUP:-$HOME/dra-switch}
K() { kubectl "$@"; }
command -v kubectl >/dev/null || K() { k3s kubectl "$@"; }

echo "== precheck: pods holding nvidia.com/gpu"
busy=$(K get pods -A -o json | python3 -c '
import json, sys
for p in json.load(sys.stdin)["items"]:
    if p["status"].get("phase") in ("Succeeded", "Failed"):
        continue
    for c in p["spec"]["containers"]:
        if "nvidia.com/gpu" in ((c.get("resources") or {}).get("limits") or {}):
            print(p["metadata"]["namespace"] + "/" + p["metadata"]["name"])')
if [ -n "$busy" ]; then echo "abort: GPU in use by $busy"; exit 1; fi

echo "== backup to $B"
mkdir -p "$B"
helm list -A > "$B/helm-list.txt"
helm get values nvdp -n nvidia-device-plugin -o yaml > "$B/nvdp-values.yaml"
K -n nvidia-device-plugin get ds -o yaml > "$B/nvdp-daemonsets.yaml"
K get node "$NODE" -o yaml > "$B/node.yaml"

echo "== stop device plugin"
K -n nvidia-device-plugin patch ds nvdp-nvidia-device-plugin --type merge \
  -p '{"spec":{"template":{"spec":{"nodeSelector":{"grp.keti.re.kr/device-plugin":"off"}}}}}'
for _ in $(seq 1 30); do
  [ "$(K -n nvidia-device-plugin get pods --no-headers 2>/dev/null | grep -vc mps-control)" = 0 ] && break
  sleep 2
done
K -n nvidia-device-plugin get pods

echo "== install NVIDIA DRA driver 0.5.0"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
helm repo update nvidia
helm upgrade --install nvidia-dra-driver-gpu nvidia/dra-driver-nvidia-gpu --version 0.5.0 \
  --create-namespace -n nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot=/ \
  --set resources.gpus.enabled=true \
  --set gpuResourcesEnabledOverride=true \
  --set resources.computeDomains.enabled=false \
  --wait --timeout 5m
K -n nvidia-dra-driver-gpu get pods -o wide

echo "== published devices"
for _ in $(seq 1 30); do K get resourceslices --no-headers 2>/dev/null | grep -q . && break; sleep 2; done
K get deviceclasses
K get resourceslices -o json | python3 -c '
import json, sys
for s in json.load(sys.stdin)["items"]:
    sp = s["spec"]
    for d in sp.get("devices", []):
        a = {k: next(iter(v.values())) for k, v in d.get("attributes", {}).items()}
        print(sp["driver"], sp.get("nodeName"), d["name"], a.get("type"), a.get("uuid"), a.get("productName"))'
K get node "$NODE" -o jsonpath='{.status.allocatable}'; echo
