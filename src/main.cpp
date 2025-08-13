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
#include "phase2/gpu_lp_boundary.h"  // GPU 리소스 정리 함수 포함
#include "utils.hpp"

// GPU 할당 함수
int allocateGPU(int mpi_rank, int mpi_size) {
    // 호스트 정보 확인
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    // MPI 통신 상태 확인
    int initialized, finalized;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    
    // 각 호스트별 rank 정보 수집
    char all_hostnames[mpi_size][256];
    
    // 배리어로 모든 프로세스가 이 지점에 도달했는지 확인
    MPI_Barrier(MPI_COMM_WORLD);
    
    // MPI_Allgather 실행
    int mpi_result = MPI_Allgather(hostname, 256, MPI_CHAR, all_hostnames, 256, MPI_CHAR, MPI_COMM_WORLD);
    
    if (mpi_result != MPI_SUCCESS) {
        std::cout << "[Rank " << mpi_rank << "] MPI_Allgather 실패: " << mpi_result << std::endl;
        return -1; // 실패
    }
    
    // 노드 내 로컬 랭크/크기 계산 (Hydra/hostfile 슬롯 없이도 안정적)
    MPI_Comm local_comm; 
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank_on_host = 0, total_ranks_on_host = 0;
    MPI_Comm_rank(local_comm, &local_rank_on_host);
    MPI_Comm_size(local_comm, &total_ranks_on_host);

    // GPU 할당: 각 서버별로 로컬 rank에 따라 GPU 할당
    int gpu_id = local_rank_on_host;
    
    // 사용 가능한 GPU 수 확인 (랭크별)
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    
    if (gpu_count > 0) gpu_id = gpu_id % gpu_count;  // GPU 수로 나눈 나머지 사용
    
    // CUDA 디바이스 설정
    cudaError_t cuda_result = cudaSetDevice(gpu_id);
    if (cuda_result != cudaSuccess) {
        std::cout << "[Rank " << mpi_rank << "] CUDA 디바이스 " << gpu_id << " 설정 실패: " 
                  << cudaGetErrorString(cuda_result) << std::endl;
        return -1;
    }
    
    // GPU 정보 확인
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    std::cout << "[Rank " << mpi_rank << "] GPU " << gpu_id << ": " << prop.name 
              << " (메모리: " << prop.totalGlobalMem / (1024*1024) << "MB, "
              << "컴퓨트 능력: " << prop.major << "." << prop.minor << ")" << std::endl;
    
    // 각 랭크의 GPU 개수 및 실제 할당 GPU ID를 수집
    std::vector<int> all_gpu_counts(mpi_size);
    MPI_Allgather(&gpu_count, 1, MPI_INT, all_gpu_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> all_assigned_gpu(mpi_size);
    MPI_Allgather(&gpu_id, 1, MPI_INT, all_assigned_gpu.data(), 1, MPI_INT, MPI_COMM_WORLD);

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
                // 각 서버별 GPU 개수에 따라 모듈러 적용해 표시
                int host_gpu_count = all_gpu_counts[rank] > 0 ? all_gpu_counts[rank] : 1;
                int assigned_gpu = all_assigned_gpu[rank];
                std::cout << "Rank" << rank << "(GPU" << assigned_gpu << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << "=====================" << std::endl;
    }
    
    MPI_Comm_free(&local_comm);
    
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
    
    // Phase1 완료 동기화
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --------------------
    // GPU 할당 (Phase2 전에 미리 수행)
    // --------------------
    if (mpi_rank == 0) {
        std::cout << "\n=== GPU 할당 ===\n";
    }
    
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