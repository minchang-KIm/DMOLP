#!/usr/bin/env bash
# run_hpc_partitioning.sh
# DMOLP 실행 자동화 (순차 실행, logs/에 태그 기반 파일명으로 로그 저장)

set -euo pipefail

############################
# 기본값(환경변수로도 덮어쓰기 가능)
############################
MPI_RUN="${MPI_RUN:-mpirun}"
EXEC="${EXEC:-/home/mpiuser/NFS/DMOLP/build/hpc_partitioning}"

# 실행 모드: bfs | random
MODE_DEFAULT="bfs"
MODE="${MODE:-$MODE_DEFAULT}"

# verbose 플래그: 0(기본) | 1
VERBOSE_DEFAULT=0
VERBOSE="${VERBOSE:-$VERBOSE_DEFAULT}"

DATASETS_DEFAULT=(
  "/home/mpiuser/NFS/DMOLP/dataset/hollywood-2011.adj-undirected.adj.txt"
  "/home/mpiuser/NFS/DMOLP/dataset/ljournal-2008.adj-undirected.adj.txt"
  "/home/mpiuser/NFS/DMOLP/dataset/wordassociation-2011.adj-undirected.adj.txt"

)

NP_LIST_DEFAULT=(2)          # 예: (2 4 8)
PARTITIONS_DEFAULT=(2 4 8 16)     # 예: (2 4 8 16)
THETAS_DEFAULT=(1000 5000 10000 50000 100000)

LOG_DIR="${OUT_DIR:-./logs}" # 로그 저장 디렉터리

################################
# 유틸
################################
parse_csv() {
  local IFS=','; read -r -a _out <<< "$1"; printf '%s\0' "${_out[@]}"
}
timestamp() { date +"%Y%m%d_%H%M%S"; }

usage() {
  cat <<'EOF'
사용법:
  run_hpc_partitioning.sh [옵션]

옵션:
  -e, --exec PATH           실행파일 경로 (기본: $EXEC)
  -n, --np LIST             mpirun 프로세스 수 CSV (예: "2,4,8")
  -d, --datasets LIST       데이터셋 경로 CSV (예: "/a.txt,/b.txt")
  -p, --partitions LIST     파티션 수 CSV (예: "2,4,8,16")
  -t, --thetas LIST         theta CSV (예: "10000,50000")
  -m, --mode MODE           phase1 모드: bfs | random (기본: bfs)
  -v, --verbose             자세한 로그(프로그램 --verbose 전달)
      --outdir DIR          로그 디렉터리 (기본 ./logs)
  -h, --help                도움말

예시:
  ./run_hpc_partitioning.sh \
    -m "random,bfs" -v \
    -d "/data/lj.txt,/data/webgraph.txt" \
    -p "2,4,8" -t "10000,50000" -n "2,4"
EOF
}

################################
# 옵션 파싱
################################
NP_LIST=("${NP_LIST_DEFAULT[@]}")
PARTITIONS=("${PARTITIONS_DEFAULT[@]}")
THETAS=("${THETAS_DEFAULT[@]}")
DATASETS=("${DATASETS_DEFAULT[@]}")

mapfile -d '' -t MODES_ARR < <(parse_csv "$MODES")

while (( "$#" )); do
  case "$1" in
    -e|--exec) EXEC="$2"; shift 2;;
    -n|--np) mapfile -d '' -t NP_LIST < <(parse_csv "$2"); shift 2;;
    -d|--datasets) mapfile -d '' -t DATASETS < <(parse_csv "$2"); shift 2;;
    -p|--partitions) mapfile -d '' -t PARTITIONS < <(parse_csv "$2"); shift 2;;
    -t|--thetas) mapfile -d '' -t THETAS < <(parse_csv "$2"); shift 2;;
    -m|--mode)
      MODES="$2"
      mapfile -d '' -t MODES_ARR < <(parse_csv "$MODES")
      shift 2
      ;;
    -v|--verbose) VERBOSE=1; shift 1;;
    --outdir) LOG_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "알 수 없는 옵션: $1"; usage; exit 1;;
  esac
done

################################
# 유효성 검사 및 준비
################################
mkdir -p "$LOG_DIR"

if [[ ! -x "$EXEC" ]]; then
  echo "실행파일이 존재하지 않거나 실행 권한이 없습니다: $EXEC" >&2
  exit 1
fi

for ds in "${DATASETS[@]}"; do
  if [[ ! -f "$ds" ]]; then
    echo "데이터셋 파일을 찾을 수 없습니다: $ds" >&2
    exit 1
  fi
done

if ! command -v "$MPI_RUN" >/dev/null 2>&1; then
  echo "mpirun(MPI) 명령을 찾을 수 없습니다: $MPI_RUN" >&2
  exit 1
fi

################################
# 실행 루틴 (순차)
################################
run_one() {
  local np="$1" ds="$2" part="$3" th="$4" mode="$5"

  local fname="${ds##*/}"
  local dataset="${fname%%.*}"             # hollywood-2011.adj-... -> hollywood-2011
  local base_ds; base_ds="$(basename "$ds")"

  local tag="${dataset}_m${mode}_np${np}_part${part}_theta${th}_$(timestamp)"
  local log_file="${LOG_DIR}/${tag}.log"

  # 프로그램 전달 옵션 구성
  local exec_opts=()
  exec_opts+=( -m "$mode" )
  if [[ "$VERBOSE" -eq 1 ]]; then
    exec_opts+=( --verbose )
  fi

  local cmd=( "$MPI_RUN" -np "$np" "$EXEC" "${exec_opts[@]}" "$ds" "$part" "$th" )

  echo "[START] ${cmd[*]}"
  {
    echo "# COMMAND : ${cmd[*]}"
    echo "# DATASET : ${ds} (basename=${base_ds})"
    echo "# MODE    : ${mode}"
    echo "# VERBOSE : ${VERBOSE}"
    echo "# LOGFILE : ${log_file}"
    echo "# START   : $(date --iso-8601=seconds)"
    "${cmd[@]}"
    rc=$?
    echo "# END     : $(date --iso-8601=seconds)"
    echo "# EXIT    : ${rc}"
    exit $rc
  } |& tee "${log_file}"
}

total=0
for mode in "${MODES_ARR[@]}"; do
  for np in "${NP_LIST[@]}"; do
    for ds in "${DATASETS[@]}"; do
      for part in "${PARTITIONS[@]}"; do
        if [[ "$mode" == "random" ]]; then
          # random: theta 무시하고 1회만 실행 (theta=1 고정)
          run_one "$np" "$ds" "$part" "1" "$mode"
          total=$(( total + 1 ))
        else
          # bfs: 전달된 theta 목록대로 모두 실행
          for th in "${THETAS[@]}"; do
            run_one "$np" "$ds" "$part" "$th" "$mode"
            total=$(( total + 1 ))
          done
        fi
      done
    done
  done
done

echo "총 ${total}개의 조합 실행을 완료했습니다."
echo "로그 디렉터리: ${LOG_DIR}"