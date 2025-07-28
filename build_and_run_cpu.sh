#!/bin/bash
set -e

# 빌드 디렉토리 생성 및 CMake configure
cmake -B build_cpu -S .

# 빌드
cmake --build build_cpu -j $(nproc)

echo "\n[빌드 완료] 실행 파일: build_cpu/dmolp_cpu"
echo "실행 예시: mpirun -np 2 ./build_cpu/dmolp_cpu <그래프파일> <파티션수>"
