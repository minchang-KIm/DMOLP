#!/bin/bash

# DMOLP CPU 버전 빌드 스크립트
# 작성일: 2025-07-22
# 설명: CPU 전용 최적화 빌드를 위한 자동 스크립트

set -e  # 에러 발생 시 즉시 종료

echo "================================================="
echo "🔧 DMOLP CPU 버전 빌드 시작"
echo "================================================="

# 프로젝트 루트 디렉토리 확인
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ 오류: CMakeLists.txt를 찾을 수 없습니다."
    echo "   프로젝트 루트 디렉토리에서 실행하세요."
    exit 1
fi

# 의존성 확인
echo "📋 의존성 확인 중..."

# MPI 확인
if ! command -v mpicc &> /dev/null; then
    echo "❌ 오류: MPI가 설치되지 않았습니다."
    echo "   설치 명령: sudo apt-get install libopenmpi-dev openmpi-bin"
    exit 1
fi

# OpenMP 확인
if ! echo '#include <omp.h>' | gcc -fopenmp -x c - -o /dev/null 2>/dev/null; then
    echo "❌ 오류: OpenMP가 지원되지 않습니다."
    echo "   GCC 4.2 이상이 필요합니다."
    exit 1
fi

echo "✅ 모든 의존성 확인 완료"

# 빌드 디렉토리 설정
BUILD_DIR="build_cpu"
if [ -d "$BUILD_DIR" ]; then
    echo "🧹 기존 빌드 디렉토리 정리 중..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "⚙️  CMake 구성 중 (CPU 최적화)..."

# CPU 최적화 CMake 설정
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -fopenmp" \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -fopenmp"

if [ $? -ne 0 ]; then
    echo "❌ CMake 구성 실패"
    exit 1
fi

echo "🔨 컴파일 중..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ 컴파일 실패"
    exit 1
fi

echo "================================================="
echo "✅ CPU 버전 빌드 완료!"
echo "================================================="
echo "📁 실행 파일: $BUILD_DIR/dmolp"
echo "🚀 실행 명령 예시:"
echo "   cd $BUILD_DIR"
echo "   mpirun -np 4 ./dmolp <graph_file> <num_partitions>"
echo ""
echo "💡 성능 팁:"
echo "   - CPU 코어 수만큼 MPI 프로세스 사용 권장"
echo "   - OpenMP 스레드 수: export OMP_NUM_THREADS=4"
echo "   - 메모리 사용량 모니터링: htop 사용"
echo "================================================="
