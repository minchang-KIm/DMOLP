#!/bin/bash
set -e

# phase2 디렉토리로 이동
dir="$(dirname "$0")/phase2"
cd "$dir"

# 빌드 디렉토리 생성 및 이동
mkdir -p build
cd build

# CMake 및 Make
cmake ..
make -j$(nproc)



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
binfile="$(pwd)/hpc_partitioning"

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
