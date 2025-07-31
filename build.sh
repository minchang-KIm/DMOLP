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

echo "[완료] phase2 빌드 성공! 실행파일: $(pwd)/mpi_distributed_workflow_v2"
