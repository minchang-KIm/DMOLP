#!/bin/bash
# DMOLP GPU-only 빌드 스크립트
# 사용법: ./build.sh

set -e

# CUDA GPU 존재 여부 확인
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi -L | grep -q GPU; then
    echo "[ERROR] CUDA GPU를 감지할 수 없습니다. GPU 환경에서만 빌드가 가능합니다."
    exit 1
fi

BUILD_DIR="build_gpu"
echo "[INFO] GPU 모드로 빌드합니다."
mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake .. -DDMOLP_USE_CUDA=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
make -j$(nproc)
echo "[SUCCESS] GPU 빌드 완료. 실행 파일: ./$PROJECT_NAME (CMakeLists.txt의 project() 명칭에 따라 다름)"

# 빌드된 실행파일을 176번 노드로 복사
scp ./build_gpu/dmolp_gpu 210.107.197.176:~/dmolp_gpu