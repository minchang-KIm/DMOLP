cmake_minimum_required(VERSION 3.10)
project(DMOLP_Phase2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CUDA (주석 해제 시 사용)
# find_package(CUDA)
# if(CUDA_FOUND)
#     include_directories(${CUDA_INCLUDE_DIRS})
#     set(CUDA_NVCC_FLAGS "-arch=sm_60")
# endif()

# 헤더 포함
include_directories(${CMAKE_SOURCE_DIR}/../Initial Partitioning)

# 소스 파일
set(SOURCES
    main.cpp
    phase1.cpp
    report_utils.cpp
)

add_executable(mpi_distributed_workflow_v2 ${SOURCES})

target_link_libraries(mpi_distributed_workflow_v2 PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
# CUDA 라이브러리 필요시 아래 주석 해제
# target_link_libraries(mpi_distributed_workflow_v2 PRIVATE ${CUDA_LIBRARIES})
