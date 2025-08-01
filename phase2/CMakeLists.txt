cmake_minimum_required(VERSION 3.18)
project(HPCPartitioning LANGUAGES CXX CUDA)

# MPI 찾기
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# CUDA 설정
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

# include 디렉토리
include_directories(${CMAKE_SOURCE_DIR}/include)

# 소스 파일
set(SOURCES
    main.cpp
    src/phase1.cpp
    src/phase2.cpp
    src/gpu_lp_boundary.cu
)

add_executable(hpc_partitioning ${SOURCES})

target_link_libraries(hpc_partitioning ${MPI_CXX_LIBRARIES} cuda cudart)

# GPU 아키텍처 (원하는 경우 수정)
set_target_properties(hpc_partitioning PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80"
)

# MPI 실행 시 빌드 후 사용법 출력
message(STATUS "Build complete. Run with:")
message(STATUS "mpirun -np <servers> ./hpc_partitioning <graph_file> <num_partitions>")