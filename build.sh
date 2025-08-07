#!/bin/bash
set -e

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
cmake ..
echo "[INFO] 컴파일 중..."
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
