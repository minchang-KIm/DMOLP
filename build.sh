#!/bin/bash
set -e

# 명령행 인수 처리
TEST_MODE=false
if [ "$1" == "--test" ]; then
    TEST_MODE=true
fi

# Intel MPI 환경 설정
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
    echo "[INFO] Intel OneAPI 환경 로드됨"
    
    # Intel 컴파일러 상태 확인
    if command -v icpc >/dev/null 2>&1; then
        echo "[INFO] Intel C++ 컴파일러 사용 가능"
        USE_INTEL_COMPILER=true
    else
        echo "[WARNING] Intel C++ 컴파일러(icpc)를 찾을 수 없음"
        USE_INTEL_COMPILER=false
    fi
elif [ -f "/opt/intel/compilers_and_libraries/linux/bin/compilervars.sh" ]; then
    source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
    echo "[INFO] Intel 컴파일러 환경 로드됨"
    
    if command -v icpc >/dev/null 2>&1; then
        echo "[INFO] Intel C++ 컴파일러 사용 가능"
        USE_INTEL_COMPILER=true
    else
        echo "[WARNING] Intel C++ 컴파일러(icpc)를 찾을 수 없음"
        USE_INTEL_COMPILER=false
    fi
else
    echo "[WARNING] Intel MPI 환경을 찾을 수 없음, OpenMPI 사용"
    USE_INTEL_COMPILER=false
fi

# OpenMP 스레드 수 자동 설정 (CPU 코어 수의 50%를 사용하여 안정성 확보)
CPU_CORES=$(nproc)
OMP_THREADS=$((CPU_CORES / 2))
if [ $OMP_THREADS -lt 1 ]; then
    OMP_THREADS=1
fi
export OMP_NUM_THREADS=$OMP_THREADS
echo "[INFO] OpenMP 스레드 수 설정: $OMP_THREADS (총 CPU 코어: $CPU_CORES)"

# 프로젝트 루트 디렉토리로 이동
dir="$(dirname "$0")"
cd "$dir"

# 기존 빌드 파일 정리
echo "[INFO] 기존 빌드 파일 정리 중..."
rm -rf build CMakeCache.txt CMakeFiles cmake_install.cmake Makefile hpc_partitioning

# 빌드 디렉토리 생성 및 이동
mkdir -p build
cd build

# CMake 및 Make (빌드 디렉토리에서 실행)
echo "[INFO] CMake 설정 중..."

# Intel 컴파일러 사용 가능 여부에 따라 설정
if [ "$USE_INTEL_COMPILER" = true ] && command -v mpiicc >/dev/null 2>&1 && command -v mpiicpc >/dev/null 2>&1; then
    echo "[INFO] Intel MPI 컴파일러 사용 (icpc + Intel MPI)"
    export CC=mpiicc
    export CXX=mpiicpc
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc ..
elif [ -n "$I_MPI_ROOT" ] && command -v mpicc >/dev/null 2>&1; then
    echo "[INFO] GNU 컴파일러 + Intel MPI 라이브러리 사용"
    export I_MPI_CC=gcc
    export I_MPI_CXX=g++
    cmake -DCMAKE_BUILD_TYPE=Release ..
else
    echo "[INFO] 표준 GNU 컴파일러 + OpenMPI 사용"
    cmake -DCMAKE_BUILD_TYPE=Release ..
fi

echo "[INFO] Release 모드로 컴파일 중 (최적화: -O3 -march=native)..."
make -j$(nproc)

# 실행파일 생성 확인 (build 디렉토리에 그대로 둠)
if [ -f "hpc_partitioning" ]; then
    echo "[INFO] 실행파일 생성 완료: $(pwd)/hpc_partitioning"
else
    echo "[ERROR] 실행파일 생성 실패"
    exit 1
fi

# 실행파일 경로를 build 디렉토리로 설정
cd ..



# 호스트파일 경로 자동 판별
if [ -f "hostfile" ]; then
    hostfile="hostfile"
elif [ -f "../hostfile" ]; then
    hostfile="../hostfile"
elif [ -f "../../phase2/hostfile" ]; then
    hostfile="../../phase2/hostfile"
elif [ -f "../../hostfile" ]; then
    hostfile="../../hostfile"
else
    echo "[ERROR] hostfile을 찾을 수 없습니다. 경로를 확인하세요."
    exit 1
fi

myip=$(hostname -I | awk '{print $1}')
binfile="$(pwd)/build/hpc_partitioning"

echo "[INFO] 실행파일을 다른 노드에 복사 중..."
while read -r line; do
    ip=$(echo "$line" | awk '{print $1}')
    if [[ "$ip" != "$myip" ]]; then
        scp "$binfile" "$ip:$binfile"
        echo "[INFO] $ip 에 복사 완료"
        sleep 2
    fi
done < "$hostfile"

echo "[완료] phase2 빌드 및 실행파일 복사 성공! 실행파일: $binfile"

# 테스트 모드 실행
if [ "$TEST_MODE" = true ]; then
    echo ""
    echo "=========================================="
    echo "           테스트 모드 실행"
    echo "=========================================="
    echo "[INFO] OpenMP 스레드 수: $OMP_NUM_THREADS"
    echo "[INFO] 실행 명령어 준비 중..."
    
    # hostfile 확인 및 생성 (검증된 Intel MPI 형식)
    intel_hostfile="./hostfile_intel.reachable"
    if [ ! -f "$intel_hostfile" ] || [ ! -s "$intel_hostfile" ]; then
        echo "[INFO] Intel MPI용 hostfile 생성"
        # OpenMPI 형식 hostfile을 Intel 형식으로 변환 (마지막 줄 개행 문제 해결)
        (cat ./hostfile; echo) | sed -E 's/[[:space:]]+slots=([0-9]+)/:\1/' | sed '/^$/d' > "$intel_hostfile"
        echo "[INFO] 생성된 hostfile 내용:"
        cat "$intel_hostfile"
    fi
    
    # 검증된 성공 환경 변수 설정
    export I_MPI_FABRICS=ofi
    export FI_PROVIDER=tcp
    export FI_TCP_IFACE=enp4s0
    
    echo "[INFO] 검증된 환경 변수 설정:"
    echo "  I_MPI_FABRICS=$I_MPI_FABRICS"
    echo "  FI_PROVIDER=$FI_PROVIDER" 
    echo "  FI_TCP_IFACE=$FI_TCP_IFACE"
    
    # 검증된 성공 명령어 사용
    MPI_CMD="/opt/intel/oneapi/mpi/2021.16/bin/mpiexec.hydra"
    HOSTFILE_ARG="$intel_hostfile"
    BINARY="$binfile"
    DATASET_PATH="./dataset/ljournal-2008.adj-undirected.adj.txt"
    
    echo "[INFO] 데이터셋을 원격 노드에 복사 중..."
    # 데이터셋 원격 복사
    myip=$(hostname -I | awk '{print $1}')
    while IFS= read -r line; do
        # 주석 제거 및 trim
        line="${line%%#*}"; line="${line%$'\r'}"; line="${line## }"; line="${line%% }"
        [ -z "$line" ] && continue
        
        # IP 추출 (Intel 형식: IP:N)
        ip="${line%%:*}"
        [ -z "$ip" ] || [ "$ip" = "$myip" ] && continue
        
        echo "[COPY] $ip: 데이터셋 디렉터리 생성 및 복사"
        ssh $SSH_OPTS "$ip" "mkdir -p '$(dirname "$DATASET_PATH")'" || true
        if [ -f "$DATASET_PATH" ]; then
            scp $SSH_OPTS "$DATASET_PATH" "$ip:$DATASET_PATH" || echo "[WARN] $ip: 데이터셋 복사 실패"
        fi
    done < "$intel_hostfile"
    
    echo "[INFO] 실행 시작..."
    echo "명령어: $MPI_CMD -hostfile $HOSTFILE_ARG -bootstrap ssh -ppn 2 -print-rank-map -l -wdir $(pwd) -n 4 $BINARY $DATASET_PATH 4 50000"
    echo ""
    
    # 검증된 성공 명령어로 실행
    $MPI_CMD -hostfile "$HOSTFILE_ARG" -bootstrap ssh -ppn 2 -print-rank-map -l -wdir "$(pwd)" -n 4 "$BINARY" "$DATASET_PATH" 4 50000
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "         실행 성공! "
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "         실행 실패 "
        echo "=========================================="
        echo "종료 코드: $exit_code"
    fi
    
    exit $exit_code
fi