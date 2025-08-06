#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <chrono>

#include "phase1/phase1.h"
#include "phase2/phase2.h"
#include "utils.hpp"

// GPU 할당 함수
int allocateGPU(int mpi_rank, int mpi_size) {
    // 호스트 정보 확인
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    std::cout << "[Rank " << mpi_rank << "] GPU 할당 시작: " << hostname << std::endl;
    std::cout.flush();
    
    // MPI 통신 상태 확인
    int initialized, finalized;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    std::cout << "[Rank " << mpi_rank << "] MPI 상태 - 초기화됨: " << initialized 
              << ", 종료됨: " << finalized << std::endl;
    std::cout.flush();
    
    // 각 호스트별 rank 정보 수집
    char all_hostnames[mpi_size][256];
    
    // 배리어로 모든 프로세스가 이 지점에 도달했는지 확인
    std::cout << "[Rank " << mpi_rank << "] MPI_Allgather 전 배리어 대기" << std::endl;
    std::cout.flush();
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "[Rank " << mpi_rank << "] 배리어 통과" << std::endl;
    std::cout.flush();
    
    std::cout << "[Rank " << mpi_rank << "] 호스트명 수집을 위한 MPI_Allgather 호출" << std::endl;
    std::cout.flush();
    
    // MPI_Allgather 실행 시간 측정
    auto allgather_start = std::chrono::high_resolution_clock::now();
    
    int mpi_result = MPI_Allgather(hostname, 256, MPI_CHAR, all_hostnames, 256, MPI_CHAR, MPI_COMM_WORLD);
    
    auto allgather_end = std::chrono::high_resolution_clock::now();
    auto allgather_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(allgather_end - allgather_start).count();
    
    if (mpi_result != MPI_SUCCESS) {
        std::cout << "[Rank " << mpi_rank << "] MPI_Allgather 실패: " << mpi_result 
                  << " (시간: " << allgather_time_ms << "ms)" << std::endl;
        std::cout.flush();
        return -1; // 실패
    }
    
    std::cout << "[Rank " << mpi_rank << "] MPI_Allgather 성공 (시간: " << allgather_time_ms << "ms)" << std::endl;
    std::cout.flush();
    
    // 수집된 호스트명 출력
    std::cout << "[Rank " << mpi_rank << "] 수집된 호스트 목록: ";
    for (int i = 0; i < mpi_size; i++) {
        std::cout << "R" << i << ":" << all_hostnames[i] << " ";
    }
    std::cout << std::endl;
    std::cout.flush();
    
    // 현재 호스트에서 몇 번째 rank인지 계산 (GPU ID로 사용)
    int local_rank_on_host = 0;
    for (int i = 0; i < mpi_rank; i++) {
        if (strcmp(hostname, all_hostnames[i]) == 0) {
            local_rank_on_host++;
        }
    }
    
    // 현재 호스트의 총 rank 수 계산
    int total_ranks_on_host = 0;
    for (int i = 0; i < mpi_size; i++) {
        if (strcmp(hostname, all_hostnames[i]) == 0) {
            total_ranks_on_host++;
        }
    }
    
    std::cout << "[Rank " << mpi_rank << "] 호스트: " << hostname 
              << " (로컬 rank: " << local_rank_on_host << "/" << total_ranks_on_host << ")" << std::endl;

    // GPU 할당: 각 서버별로 로컬 rank에 따라 GPU 할당
    int gpu_id = local_rank_on_host;
    
    // 사용 가능한 GPU 수 확인
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    
    if (gpu_id >= gpu_count) {
        std::cout << "[Rank " << mpi_rank << "] 경고: GPU " << gpu_id 
                  << " 요청했지만 사용 가능한 GPU는 " << gpu_count << "개입니다." << std::endl;
        gpu_id = gpu_id % gpu_count;  // GPU 수로 나눈 나머지 사용
        std::cout << "[Rank " << mpi_rank << "] GPU " << gpu_id << "로 변경합니다." << std::endl;
    }
    
    // CUDA 디바이스 설정
    cudaError_t cuda_result = cudaSetDevice(gpu_id);
    if (cuda_result != cudaSuccess) {
        std::cout << "[Rank " << mpi_rank << "] CUDA 디바이스 " << gpu_id << " 설정 실패: " 
                  << cudaGetErrorString(cuda_result) << std::endl;
        return -1;
    } else {
        std::cout << "[Rank " << mpi_rank << "] CUDA 디바이스 " << gpu_id << " 설정 성공" << std::endl;
    }
    
    // GPU 정보 확인
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    std::cout << "[Rank " << mpi_rank << "] GPU " << gpu_id << ": " << prop.name 
              << " (메모리: " << prop.totalGlobalMem / (1024*1024) << "MB, "
              << "컴퓨트 능력: " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout.flush();
    
    // 전체 클러스터의 GPU 배치 정보 출력 (Rank 0에서만)
    if (mpi_rank == 0) {
        std::cout << "\n=== GPU 배치 정보 ===" << std::endl;
        std::map<std::string, std::vector<int>> host_to_ranks;
        
        // 호스트별 rank 정리
        for (int i = 0; i < mpi_size; i++) {
            host_to_ranks[std::string(all_hostnames[i])].push_back(i);
        }
        
        for (const auto& entry : host_to_ranks) {
            std::cout << "호스트 " << entry.first << ": ";
            for (int rank : entry.second) {
                // 해당 호스트에서의 로컬 rank 계산
                int local_rank_calc = 0;
                for (int j = 0; j < rank; j++) {
                    if (strcmp(entry.first.c_str(), all_hostnames[j]) == 0) {
                        local_rank_calc++;
                    }
                }
                // 각 서버별로 독립적인 GPU 할당
                int assigned_gpu = local_rank_calc;
                std::cout << "Rank" << rank << "(GPU" << assigned_gpu << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << "=====================" << std::endl;
    }
    
    // 모든 프로세스 동기화
    MPI_Barrier(MPI_COMM_WORLD);
    
    return gpu_id;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc < 4) {
        if (mpi_rank == 0)
            std::cout << "Usage: mpirun -np <servers> ./hpc_partitioning <graph_file> <num_partitions> <theta> \n";
        MPI_Finalize();
        return 1;
    }

    const char* graph_file = argv[1];
    const int num_partitions = atoi(argv[2]);
    const int theta = atoi(argv[3]);

    // MPI 통신 테스트
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    std::cout << "[Rank " << mpi_rank << "/" << mpi_size << "] 시작 - 호스트: " << hostname << std::endl;
    std::cout.flush();
    
    // 간단한 통신 테스트
    int test_value = mpi_rank;
    std::vector<int> all_ranks(mpi_size);
    
    std::cout << "[Rank " << mpi_rank << "] 초기 통신 테스트 시작" << std::endl;
    std::cout.flush();
    
    MPI_Allgather(&test_value, 1, MPI_INT, all_ranks.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    std::cout << "[Rank " << mpi_rank << "] 초기 통신 테스트 성공 - 수집된 rank들: ";
    for (int i = 0; i < mpi_size; i++) {
        std::cout << all_ranks[i] << " ";
    }
    std::cout << std::endl;
    std::cout.flush();

    
    Graph local_graph;
    GhostNodes ghost_nodes;

    // --------------------
    // Phase 1 실행
    // --------------------
    if (mpi_rank == 0) {
        std::cout << "\n=== Phase 1: Initial Partitioning & Distribution ===\n";
    }

    Phase1Metrics metrics1_raw = run_phase1(
        mpi_rank, mpi_size,
        graph_file,
        num_partitions,
        theta,
        local_graph,
        ghost_nodes
    );

    
    PartitioningMetrics metrics1(metrics1_raw, num_partitions);
    
    std::cout << "[Rank " << mpi_rank << "] Phase1 완료, GPU 할당 단계로 진입" << std::endl;
    std::cout.flush();
    
    // Phase1 완료 동기화 - 더 자세한 로그
    std::cout << "[Rank " << mpi_rank << "] Phase1 완료 배리어 진입 시도" << std::endl;
    std::cout.flush();
    
    auto barrier_start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    auto barrier_end = std::chrono::high_resolution_clock::now();
    auto barrier_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(barrier_end - barrier_start).count();
    
    std::cout << "[Rank " << mpi_rank << "] Phase1 완료 배리어 통과 (대기시간: " << barrier_time_ms << "ms)" << std::endl;
    std::cout.flush();
    
    // --------------------
    // GPU 할당 (Phase2 전에 미리 수행)
    // --------------------
    if (mpi_rank == 0) {
        std::cout << "\n=== GPU 할당 ===\n";
    }
    
    std::cout << "[Rank " << mpi_rank << "] GPU 할당 함수 호출 시작" << std::endl;
    std::cout.flush();
    
    int gpu_id = allocateGPU(mpi_rank, mpi_size);
    if (gpu_id < 0) {
        std::cout << "[Rank " << mpi_rank << "] GPU 할당 실패, 종료합니다." << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    if (mpi_rank == 0) {
        std::cout << "\n=== Phase 2: Label Propagation ===\n";
    }
    
    // --------------------
    // Phase 2 실행
    // --------------------

    PartitioningMetrics metrics2 = run_phase2(
        mpi_rank, mpi_size,
        num_partitions,
        local_graph,
        ghost_nodes,
        gpu_id  // GPU ID 전달
    );

    if (mpi_rank == 0) {
        printComparisonReport(metrics1, metrics2);
    }
    
    MPI_Finalize();
    return 0;
}