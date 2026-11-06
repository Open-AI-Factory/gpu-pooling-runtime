#!/usr/bin/env bash
cd "$(dirname "$0")/.."
VENV=/root/venvs/grp
LOG=/root/operator.log
case "${1:-start}" in
  stop)  pkill -f "^$VENV/bin/python3? $VENV/bin/kopf" ; sleep 1 ;;
  start) pkill -f "^$VENV/bin/python3? $VENV/bin/kopf" ; sleep 1
         nohup $VENV/bin/kopf run -m grp_runtime.runtime_coordination.runtime_action_controller.operator --all-namespaces > $LOG 2>&1 &
         sleep 4; echo "operator pid $(pgrep -f "^$VENV/bin/python3? $VENV/bin/kopf")" ;;
  log)   grep -E "grp_runtime|ERROR|Traceback|Error" $LOG | tail -${2:-40} ;;
esac
