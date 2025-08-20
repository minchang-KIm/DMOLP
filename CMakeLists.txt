cmake_minimum_required(VERSION 3.18)

# ─────────────────────────────────────────────────────────────
# (권장) NVCC 11.5 + GCC 11 조합 이슈 회피: GCC 10 고정
# 가능하다면 아래 주석 해제 후 사용하십시오.
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)
# ─────────────────────────────────────────────────────────────

project(HPCPartitioning LANGUAGES CXX CUDA)

# 패키지 탐색
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

message(STATUS "MPI C compiler: ${MPI_C_COMPILER}")
message(STATUS "MPI CXX compiler: ${MPI_CXX_COMPILER}")
message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")

# 표준 지정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 기본 빌드 타입
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 최적화 플래그
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DDEBUG")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG   "-O0 -g -DDEBUG")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

# CRoaring 경로
set(CROARING_PATH "/home/intern_graph/croaring_install")

# 실행 파일
add_executable(hpc_partitioning
  src/main.cpp
  src/utils.cpp
  src/phase1/phase1.cpp
  src/phase1/init.cpp
  src/phase1/partition.cpp
  src/phase1/seed.cpp
  src/phase1/random.cpp
  src/phase2/phase2.cpp
  src/phase2/gpu_lp_boundary.cu
)

# include 디렉토리
target_include_directories(hpc_partitioning PRIVATE
  ${MPI_INCLUDE_PATH}
  ${CMAKE_SOURCE_DIR}/include
  ${CROARING_PATH}/include
)

# 컴파일 정의
target_compile_definitions(hpc_partitioning PRIVATE
  GPU_PARTITION
  USE_MASTER_WORKER_PENALTY
  USE_PINNED_MEMORY_OPTIMIZATION
)

# OpenMP
if(OpenMP_CXX_FOUND)
  target_link_libraries(hpc_partitioning PRIVATE OpenMP::OpenMP_CXX)
endif()

# CUDA 옵션 (디바이스 람다 대비 및 라인정보)
target_compile_options(hpc_partitioning PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  # (GCC 10 강제 필요 시) 다음 줄 주석 해제
  # $<$<COMPILE_LANGUAGE:CUDA>:-ccbin=/usr/bin/g++-10>
)

# 아키텍처 (RTX 3080 = sm_86 포함)
set_target_properties(hpc_partitioning PROPERTIES
  CUDA_ARCHITECTURES "60;70;75;80;86"
)

# 최종 링크를 C++ 링커로 강제 + as-needed 방지
set_target_properties(hpc_partitioning PROPERTIES
  LINKER_LANGUAGE CXX
)

target_link_options(hpc_partitioning PRIVATE
  -Wl,--no-as-needed
)

# 라이브러리 링크 (공식 타깃 사용 권장)
target_link_libraries(hpc_partitioning PRIVATE
  MPI::MPI_CXX
  ${CROARING_PATH}/lib/libroaring.a
  CUDA::cudart
  CUDA::cuda_driver
  stdc++           # 링커가 C++ 표준 라이브러리를 놓치지 않도록 명시
  # (filesystem 사용 시) stdc++fs 필요할 수 있음:
  # stdc++fs
)

# 정적 libstdc++/libgcc는 충돌 소지가 있어 제거 권장
# set_target_properties(hpc_partitioning PROPERTIES
#   LINK_FLAGS "-static-libgcc -static-libstdc++"
# )

# RPATH (환경에 맞게 조정)
set_target_properties(hpc_partitioning PROPERTIES
  INSTALL_RPATH "/lib/x86_64-linux-gnu;/usr/lib/x86_64-linux-gnu"
  BUILD_WITH_INSTALL_RPATH TRUE
)

message(STATUS "Build complete. Run with:")
message(STATUS "mpirun -np <servers> ./hpc_partitioning <graph_file> <num_partitions>")
