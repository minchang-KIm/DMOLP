#!/bin/bash

# DMOLP GPU (CUDA) 버전 빌드 스크립트
# 작성일: 2025-07-22
# 설명: CUDA GPU 가속을 활용한 고성능 빌드를 위한 자동 스크립트

set -e  # 에러 발생 시 즉시 종료

echo "================================================="
echo "🚀 DMOLP GPU (CUDA) 버전 빌드 시작"
echo "================================================="

# 프로젝트 루트 디렉토리 확인
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ 오류: CMakeLists.txt를 찾을 수 없습니다."
    echo "   프로젝트 루트 디렉토리에서 실행하세요."
    exit 1
fi

# GPU 및 CUDA 확인
echo "🔍 GPU 및 CUDA 환경 확인 중..."

# NVIDIA GPU 확인
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 오류: NVIDIA GPU 또는 드라이버를 찾을 수 없습니다."
    echo "   nvidia-smi 명령이 작동하지 않습니다."
    exit 1
fi

# CUDA 컴파일러 확인
if ! command -v nvcc &> /dev/null; then
    echo "❌ 오류: CUDA 컴파일러(nvcc)를 찾을 수 없습니다."
    echo "   CUDA Toolkit이 설치되지 않았습니다."
    echo "   설치 가이드: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# GPU 정보 출력
echo "🖥️  GPU 정보:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -1

# CUDA 버전 확인
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
echo "🔧 CUDA 버전: $CUDA_VERSION"

if [ $(echo "$CUDA_VERSION < 11.0" | bc -l) -eq 1 ]; then
    echo "⚠️  경고: CUDA 11.0 이상 권장 (현재: $CUDA_VERSION)"
fi

# MPI 확인
if ! command -v mpicc &> /dev/null; then
    echo "❌ 오류: MPI가 설치되지 않았습니다."
    echo "   설치 명령: sudo apt-get install libopenmpi-dev openmpi-bin"
    exit 1
fi

echo "✅ 모든 의존성 확인 완료"

# 빌드 디렉토리 설정
BUILD_DIR="build_gpu"
if [ -d "$BUILD_DIR" ]; then
    echo "🧹 기존 빌드 디렉토리 정리 중..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "⚙️  CMake 구성 중 (GPU 가속 활성화)..."

# GPU 최적화 CMake 설정
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -fopenmp" \
    -DCMAKE_CUDA_FLAGS="-O3 -use_fast_math --expt-relaxed-constexpr" \
    -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86"

if [ $? -ne 0 ]; then
    echo "❌ CMake 구성 실패"
    echo "💡 팁: CUDA Toolkit 경로를 확인하세요"
    echo "   export PATH=/usr/local/cuda/bin:\$PATH"
    echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    exit 1
fi

echo "🔨 컴파일 중 (GPU 커널 포함)..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ 컴파일 실패"
    echo "💡 일반적인 해결 방법:"
    echo "   1. CUDA 아키텍처 호환성 확인"
    echo "   2. GPU 메모리 부족 시 --maxrregcount 옵션 추가"
    echo "   3. CUDA 드라이버 업데이트"
    exit 1
fi

echo "================================================="
echo "✅ GPU 버전 빌드 완료!"
echo "================================================="
echo "📁 실행 파일: $BUILD_DIR/dmolp"
echo "🚀 실행 명령 예시:"
echo "   cd $BUILD_DIR"
echo "   mpirun -np 2 ./dmolp <graph_file> <num_partitions>"
echo ""
echo "🎮 GPU 최적화 팁:"
echo "   - GPU 메모리 사용량 모니터링: nvidia-smi -l 1"
echo "   - CUDA 스트림 수: 기본값 4개 (코드에서 조정 가능)"
echo "   - 블록 크기: 256 스레드 (Tesla V100 최적화)"
echo "   - 메모리 접합: 연속 메모리 접근 패턴 사용"
echo ""
echo "⚡ 성능 비교:"
echo "   - CPU 대비 예상 가속비: 10-50x (그래프 크기에 따라)"
echo "   - 메모리 대역폭 활용률: ~80% (최적화된 경우)"
echo "================================================="
