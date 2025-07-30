#!/bin/bash
# DMOLP 전체 환경 자동 설치 스크립트 (Miniconda + RAPIDS + 의존성, conda init/activate 포함)
# 사용법: bash install.sh

set -e

# 1. Miniconda 설치 (이미 설치되어 있으면 건너뜀)
if ! command -v conda &> /dev/null; then
    echo "[INFO] Miniconda 설치 중..."
    MINICONDA=Miniconda3-latest-Linux-x86_64.sh
    wget -q https://repo.anaconda.com/miniconda/$MINICONDA -O /tmp/$MINICONDA
    if [ -d "$HOME/miniconda" ]; then
        echo "[INFO] 기존 Miniconda 디렉터리 삭제 후 재설치합니다."
        rm -rf "$HOME/miniconda"
    fi
    bash /tmp/$MINICONDA -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc
    hash -r
else
    echo "[INFO] Miniconda가 이미 설치되어 있습니다."
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# 2. conda init (bash 쉘에 conda 명령어 등록)
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init bash

# 3. conda 환경 생성 및 활성화
if ! conda info --envs | grep -q rapids-25; then
    echo "[INFO] conda rapids-25 환경 생성 중..."
    conda create -y -n rapids-25 python=3.10
fi
conda activate rapids-25

# 4. RAPIDS 및 빌드 도구 설치
conda install -y -c rapidsai -c conda-forge -c nvidia \
    cugraph=25.06 \
    rmm=25.06 \
    raft-dask=25.06 \
    openmpi pkg-config cmake make gcc_linux-64 gxx_linux-64

cat <<EOF

[설치 완료]

- conda 환경: rapids-25 (자동 활성화됨)
- RAPIDS 라이브러리(cugraph, rmm, raft-dask) 및 빌드 도구 설치 완료
- CMake 빌드 시 아래 옵션 사용:
    cmake -DCMAKE_PREFIX_PATH="\$CONDA_PREFIX" ..
- 실행 예시:
    mpirun -np 4 ./dmolp_gpu <그래프파일> <파티션수>

[참고] CUDA 드라이버 및 GPU 환경이 정상적으로 동작해야 하며, 설치 경로에 따라 CMake 옵션을 조정하세요.
EOF