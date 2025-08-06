cmake_minimum_required(VERSION 3.18)
project(HPCPartitioning LANGUAGES CXX CUDA)

# MPI, OpenMP, CUDA 찾기
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# CRoaring 라이브러리 설정 (새로 설치한 버전) - 정적 링킹
set(CROARING_PATH "/home/intern_graph/croaring_install")
include_directories(${CROARING_PATH}/include)
set(ROARING_LIBRARY "${CROARING_PATH}/lib/libroaring.a")

# CUDA 설정
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

# include 디렉토리
include_directories(${CMAKE_SOURCE_DIR}/include)

# 소스 파일
set(SOURCES
    src/main.cpp
    src/utils.cpp
    src/phase1/phase1.cpp
    src/phase1/init.cpp
    src/phase1/partition.cpp
    src/phase1/seed.cpp
    src/phase2/phase2.cpp
    src/phase2/gpu_lp_boundary.cu
)

add_executable(hpc_partitioning ${SOURCES})

# 링킹할 라이브러리들
target_link_libraries(hpc_partitioning 
    ${MPI_CXX_LIBRARIES} 
    OpenMP::OpenMP_CXX 
    ${ROARING_LIBRARY}
    cuda 
    cudart
)

# 정적 링킹 설정
set_target_properties(hpc_partitioning PROPERTIES
    LINK_FLAGS "-static-libgcc -static-libstdc++"
)

# RPATH 설정 (실행 시 라이브러리 경로)
set_target_properties(hpc_partitioning PROPERTIES
    INSTALL_RPATH "/lib/x86_64-linux-gnu;/usr/lib/x86_64-linux-gnu"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# GPU 아키텍처 (원하는 경우 수정)
set_target_properties(hpc_partitioning PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80"
)

# MPI 실행 시 빌드 후 사용법 출력
message(STATUS "Build complete. Run with:")
message(STATUS "mpirun -np <servers> ./hpc_partitioning <graph_file> <num_partitions>")