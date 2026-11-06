#!/usr/bin/env bash
set -euo pipefail
ROLE=${1:?usage: share-checkpoints.sh server|client}
DIR=${DIR:-/opt/grp-runtime/runs/k8s}
SERVER=${SERVER:?set SERVER to the NFS server address}
CLIENTS=${CLIENTS:?set CLIENTS to the client addresses}
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

if [ "$ROLE" = server ]; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nfs-kernel-server >/dev/null
  mkdir -p "$DIR"
  sed -i "\#^$DIR #d" /etc/exports
  echo "$DIR $(for c in $CLIENTS; do printf '%s(rw,sync,no_subtree_check,no_root_squash) ' "$c"; done)" >> /etc/exports
  exportfs -ra
  systemctl enable --now nfs-server
  exportfs -v
else
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nfs-common >/dev/null
  mkdir -p "$DIR"
  sed -i "\#[[:space:]]$DIR[[:space:]]#d" /etc/fstab
  echo "$SERVER:$DIR $DIR nfs4 rw,hard,_netdev 0 0" >> /etc/fstab
  systemctl daemon-reload
  mount "$DIR"
  findmnt "$DIR"
fi
