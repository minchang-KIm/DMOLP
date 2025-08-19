#!/bin/bash
set -e

# --- 스크립트 설정 ---
# 테스트 모드 플래그
TEST_MODE=false
if [ "$1" == "--test" ]; then
    TEST_MODE=true
    echo "=========================================="
    echo "           테스트 모드 실행"
    echo "=========================================="
fi

# --- 1. 빌드 환경 설정 및 컴파일 ---

# OpenMP 스레드 수 자동 설정 (서버 부하를 줄이기 위해 코어 수의 절반 사용)
CPU_CORES=$(nproc)
OMP_THREADS=$((CPU_CORES / 2))
[ $OMP_THREADS -lt 1 ] && OMP_THREADS=1
export OMP_NUM_THREADS=$OMP_THREADS
echo "[INFO] OpenMP 스레드 수를 $OMP_THREADS 으로 설정했습니다. (총 코어: $CPU_CORES)"

# 프로젝트 루트 디렉토리로 이동
cd "$(dirname "$0")"

# 기존 빌드 파일 정리
echo "[INFO] 이전 빌드 파일을 정리합니다..."
rm -rf build

# 빌드 디렉토리 생성 및 CMake 실행
mkdir -p build
cd build
echo "[INFO] CMake 설정을 시작합니다..."
cmake .. -DCMAKE_BUILD_TYPE=Release
echo "[INFO] Release 모드로 컴파일을 시작합니다..."
make -j$(nproc)

# 실행 파일 확인
EXECUTABLE_NAME="hpc_partitioning"
if [ ! -f "$EXECUTABLE_NAME" ]; then
    echo "[ERROR] 실행 파일($EXECUTABLE_NAME) 생성에 실패했습니다."
    exit 1
fi
echo "[SUCCESS] 빌드 성공! 실행 파일: $(pwd)/$EXECUTABLE_NAME"
cd .. # 다시 프로젝트 루트로 이동

# --- 2. 실행 파일 배포 ---

# 호스트 파일 경로 찾기
HOSTFILE="hostfile"
if [ ! -f "$HOSTFILE" ]; then
    echo "[ERROR] '$HOSTFILE'을 찾을 수 없습니다. 프로젝트 루트 디렉토리에 있는지 확인하세요."
    exit 1
fi

# 모든 서버의 ~/bin 디렉토리에 실행 파일 복사
echo "[INFO] 모든 서버에 실행 파일을 배포합니다..."
LOCAL_EXECUTABLE_PATH="$(pwd)/build/$EXECUTABLE_NAME"
REMOTE_EXECUTABLE_PATH="~/bin/$EXECUTABLE_NAME"

# 서버 목록을 읽어옴 (주석 및 빈 줄 제외)
HOSTS=$(grep -vE '^\s*#|^\s*$' "$HOSTFILE" | awk '{print $1}')

for host in $HOSTS; do
    echo "[DEPLOY] -> $host 서버로 복사 중..."
    # 원격 서버에 ~/bin 디렉토리 생성 (없을 경우)
    ssh "$host" "mkdir -p ~/bin"
    # 실행 파일 복사
    scp "$LOCAL_EXECUTABLE_PATH" "$host:$REMOTE_EXECUTABLE_PATH"
    # 실행 권한 부여
    ssh "$host" "chmod +x $REMOTE_EXECUTABLE_PATH"
done
echo "[SUCCESS] 모든 서버에 실행 파일 배포 완료!"


# --- 3. 테스트 모드 실행 ---

if [ "$TEST_MODE" = true ]; then
    echo ""
    echo "[INFO] 테스트 실행을 준비합니다..."

    # Intel MPI용 hostfile 생성 (IP:slots 형식)
    INTEL_HOSTFILE="./hostfile_intel.run"
    grep -vE '^\s*#|^\s*$' "$HOSTFILE" | sed -E 's/[[:space:]]+slots=([0-9]+)?/:2/' > "$INTEL_HOSTFILE"
    echo "[INFO] Intel MPI용 임시 hostfile 생성:"
    cat "$INTEL_HOSTFILE"
    
    # Intel MPI 환경 변수 설정
    export I_MPI_FABRICS=ofi
    export FI_PROVIDER=tcp
    export FI_TCP_IFACE=enp4s0 # 사용자의 네트워크 인터페이스 이름
    
    # 데이터셋 경로 및 원격 복사
    DATASET_PATH="./dataset/ljournal-2008.adj-undirected.adj.txt"
    echo "[INFO] 원격 서버에 데이터셋을 동기화합니다..."
    for host in $HOSTS; do
        if [ "$host" != "$(hostname)" ] && [ "$host" != "$(hostname -i)" ]; then
             echo "[COPY] -> $host 서버로 복사 중..."
             ssh "$host" "mkdir -p '$(dirname "$DATASET_PATH")'"
             scp "$DATASET_PATH" "$host:$DATASET_PATH"
        fi
    done

    # MPI 실행
    echo "[INFO] MPI 프로그램을 시작합니다..."
    
    # 명령어 변수화
    MPI_EXEC="/opt/intel/oneapi/mpi/2021.16/bin/mpiexec.hydra"
    MPI_OPTIONS="-hostfile $INTEL_HOSTFILE -bootstrap ssh -ppn 2 -n 4"
    WDIR_OPTION="-wdir $(pwd)"
    
    # 실행 파일의 경로를 원격 서버에 배포된 경로로 지정합니다.
    REMOTE_EXECUTABLE_PATH="~/bin/$EXECUTABLE_NAME"

    # 실행할 최종 명령어 출력
    echo "--------------------------------------------------"
    echo "EXECUTING: $MPI_EXEC $MPI_OPTIONS $WDIR_OPTION $REMOTE_EXECUTABLE_PATH $DATASET_PATH 4 50000"
    echo "--------------------------------------------------"

    # 수정한 경로로 실행
    $MPI_EXEC $MPI_OPTIONS $WDIR_OPTION "$REMOTE_EXECUTABLE_PATH" "$DATASET_PATH" 4 50000

    exit_code=$?
    echo "=========================================="
    if [ $exit_code -eq 0 ]; then
        echo "실행 성공!"
    else
        echo "실행 실패 (종료 코드: $exit_code)"
    fi
    echo "=========================================="
    exit $exit_code
fi