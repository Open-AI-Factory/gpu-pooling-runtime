#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/venvs/grp/bin:$PATH
ROOTDIR=runs/capture
REQ=${PLAN_REQUEST:-scenarios/plan-request/pr-train-kind.yaml}
DIR=${CAPTURE_DIR:-$(ls -d $ROOTDIR/*/ 2>/dev/null | sort | tail -1)}
DIR=${DIR%/}

run() { printf '$ %s\n' "$1"; bash -c "$1" 2>&1; echo; }
note() { printf '# %s\n' "$*"; }
wait_pr() { for _ in $(seq 1 30); do [ -n "$(kubectl get planrequest train-plan -o jsonpath='{.status.result}' 2>/dev/null)" ] && return 0; sleep 1; done; return 1; }
wait_ra() { for _ in $(seq 1 120); do p=$(kubectl get ra "$1" -o jsonpath='{.status.phase}' 2>/dev/null)
            [ "$p" = Succeeded ] || [ "$p" = Aborted ] && return 0; sleep 2; done; return 1; }
wait_running() {
  for _ in $(seq 1 90); do
    [ "$(kubectl get eplan train-plan -o jsonpath='{.status.state}' 2>/dev/null)" = ReadyToRun ] && \
      kubectl get cm grp-train -o jsonpath='{.data.state}' 2>/dev/null | grep -q '"status": "running"' && return 0
    sleep 2
  done; echo "!! workload not running"; return 1; }
watch_start() { keticti runtime watch -v --timeout 300 > "$DIR/.watch.txt" 2>&1 </dev/null & echo $! > "$DIR/.watch.pid"; sleep 2; }
watch_show() { sleep 3; kill "$(cat "$DIR/.watch.pid" 2>/dev/null)" 2>/dev/null; rm -f "$DIR/.watch.pid"
               echo '$ keticti runtime watch -v   (별도 터미널)'; cat "$DIR/.watch.txt"; echo; }
transitions() { run "kubectl get ra $1 -o jsonpath='{range .status.conditions[*]}{.phase}{\"\\t\"}{.result}{\"\\n\"}{end}'"; }

step() {
  local s=$1; [ -n "$DIR" ] || { echo "run 'prep' first"; exit 1; }
  case $s in
  A)  note "캡처 A: 외부 실행 요청 (PlanRequest)"
      run "grep -v '^#' $REQ"
      run "grep -nE '^ +(parallelism|allocation):|gpusPerNode:' $REQ || echo '(parallelism·gpusPerNode·allocation 없음: GPU 수만 고정 입력, 워커 구성·DP×TP×PP·노드는 지정하지 않음)'"
      run "kubectl apply -f $REQ"
      wait_pr
      run "kubectl get planrequest train-plan" ;;
  B)  note "캡처 B: 실행 요구사항 분석·병렬화 후보 (자원 풀 토폴로지 요약 기준, 특정 GPU·노드는 고르지 않음, 규칙 기반 선정)"
      note "GPU 수는 요청의 고정 입력 (analyzed=false, 자동 산정은 미구현)"
      run "kubectl get planrequest train-plan -o jsonpath='{.status.gpuCount}'"
      run "kubectl get planrequest train-plan -o jsonpath='{.status.topology}'"
      run "kubectl get planrequest train-plan -o jsonpath='nodes={.status.workers.nodes} gpusPerNode={.status.workers.gpusPerNode} (from {.status.workers.gpusPerNodeSource})'"
      run "kubectl get planrequest train-plan -o jsonpath='{range .status.candidates[*]}{@}{\"\\n\"}{end}'"
      run "kubectl get planrequest train-plan -o jsonpath='{.status.selectionRule}'"
      run "kubectl get planrequest train-plan -o jsonpath='selected={.status.selected}{\"\\n\"}reason={.status.decisionReason}'" ;;
  C)  note "캡처 C: ExecutionPlan 자동 생성 → 스케줄러(scheduler-mock) 토폴로지 기반 배치 → rank Pod"
      wait_running
      run "kubectl get eplan train-plan"
      note "① ExecutionPlan: 워커 구성 요구만 있고 노드 이름 없음"
      run "kubectl get eplan train-plan -o jsonpath='requestRef={.spec.planning.requestRef}{\"\\n\"}parallelism={.spec.parallelism}{\"\\n\"}placement={.spec.placement}{\"\\n\"}rankSlots={.spec.planning.workers.rankSlots}{\"\\n\"}constraints={.spec.planning.workers.constraints}'"
      note "② 스케줄러 배치 결과 (계획 생성 뒤)"
      run "kubectl get eplan train-plan -o jsonpath='{.status.placement}'"
      run "kubectl get planrequest train-plan"
      note "③ rank 실행·격리값"
      run "kubectl get eplan train-plan -o jsonpath='world={.status.world} assignedNodes={.status.assignedNodes}{\"\\n\"}{range .status.isolation.requested[*]}rank {.rank}: node={.node} cpu={.cpuLimit} memoryMi={.memoryLimitMb} device={.device}{\"\\n\"}{end}'"
      run "kubectl get pods -o wide -l grp.keti.re.kr/workload=train"
      run "keticti runtime agents train -v" ;;
  H)  note "캡처 H: 스케줄러 배치 불가 — 계획은 유효(Accepted)하지만 노드 3개 필요, Kind worker 2개 → Unschedulable, 실행하지 않고 대기"
      run "kubectl apply -f scenarios/execution-plan/ep-unschedulable.yaml"
      sleep 5
      run "kubectl get eplan"
      run "kubectl get eplan big-plan -o jsonpath='validation={.status.validation} state={.status.state}{\"\\n\"}placement={.status.placement}'"
      run "kubectl get pods -l grp.keti.re.kr/workload=train-big"
      run "kubectl delete eplan big-plan" ;;
  D)  note "캡처 D: RuntimeAction 요청 및 Worker 준비 상태"
      wait_running
      run "keticti runtime agents train"
      watch_start
      run "keticti quota set train --gpus 1 --name resize-01"
      run "kubectl get ra resize-01 -o jsonpath='{.spec}'"
      wait_ra resize-01
      run "kubectl get ra resize-01" ;;
  E)  note "캡처 E: 정상 처리 상태 전이 (resize-01, GPU 2→1)"
      watch_show
      transitions resize-01
      sleep 4; run "keticti runtime agents train" ;;
  F)  note "캡처 F: 준비 기한 초과 — rank 0 일시정지(응답 없음 재현) → 변경 전 Aborted, 기존 구성 유지"
      wait_running
      run "keticti debug pause train --rank 0"
      note "heartbeat(10s) 만료 대기"; sleep 13
      run "keticti runtime agents train"
      watch_start
      run "keticti quota set train --gpus 2 --deadline 15 --name resize-02"
      wait_ra resize-02
      watch_show
      transitions resize-02
      run "kubectl get ra resize-02 -o jsonpath='reason={.status.reason} previous={.status.previous}{\"\\n\"}{.status.message}'"
      run "keticti debug resume train --rank 0"
      sleep 8; run "keticti runtime agents train" ;;
  G)  note "캡처 G: 재구성 실패 — 재시작 단계에 실패 주입(없는 checkpoint) → 기존 구성으로 복구"
      wait_running
      watch_start
      run "keticti quota set train --gpus 2 --name resize-03 --inject-failure"
      wait_ra resize-03
      watch_show
      transitions resize-03
      run "kubectl get ra resize-03 -o jsonpath='reason={.status.reason} previous={.status.previous}{\"\\n\"}{.status.message}'"
      sleep 4; run "keticti runtime agents train"
      run "kubectl get eplan train-plan -o jsonpath='{range .status.history[*]}gen {.generation}: {.event} {.parallelism}{\"\\n\"}{end}'"
      run "kubectl get ra" ;;
  esac
}

case "${1:-}" in
  prep)
    DIR=$ROOTDIR/$(date +%Y%m%d-%H%M%S); mkdir -p "$DIR"
    kubectl apply -f deploy/crd-planrequest.yaml -f deploy/crd-executionplan.yaml -f deploy/crd-runtimeaction.yaml >/dev/null
    bash deploy/operator.sh start
    bash deploy/slide15-scenario.sh reset
    echo "capture dir: $DIR" ;;
  A|B|C|D|E|F|G|H)
    step "$1" | tee "$DIR/$1.txt" ;;
  all)
    bash "$0" prep
    DIR=$(ls -d $ROOTDIR/*/ | sort | tail -1); DIR=${DIR%/}
    for s in A B C D E F G H; do echo; echo "==================== $s"; step $s | tee "$DIR/$s.txt"; done
    rm -f "$DIR/.watch.txt"
    echo; echo "saved: $DIR" ;;
  *) sed -n '2,10p' "$0" ;;
esac
