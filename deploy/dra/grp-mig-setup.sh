#!/bin/sh
set -eu
GPU=${GPU:-0}
PROFILES=${PROFILES:-14,14,14,14}
if [ "$(nvidia-smi -i "$GPU" --query-gpu=mig.mode.current --format=csv,noheader)" != "Enabled" ]; then
  nvidia-smi -i "$GPU" -mig 1
fi
if ! nvidia-smi mig -i "$GPU" -lgi | grep -q "MIG "; then
  nvidia-smi mig -i "$GPU" -cgi "$PROFILES" -C
fi
nvidia-smi -L
