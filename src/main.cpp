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

struct Options {
    bool verbose = false;
    bool mode = false; // false => bfs / true => random
    const char* graph_file = nullptr;
    int num_partitions = 0;
    int theta = 0;
};

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

static bool parse_args(int argc, char** argv, Options& opts, int mpi_rank) {
    std::vector<const char*> positionals;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];

        // 종료 토큰
        if (std::strcmp(a, "--") == 0) {
            for (int j = i + 1; j < argc; ++j) positionals.push_back(argv[j]);
            break;
        }

        // --verbose / -v
        if (std::strcmp(a, "--verbose") == 0 || std::strcmp(a, "-v") == 0) {
            opts.verbose = true;
            continue;
        }

        // -m <mode>
        if (std::strcmp(a, "-m") == 0) {
            if (i + 1 >= argc) {
                if (mpi_rank == 0) std::cerr << "오류: -m 인자에 값이 없습니다.\n";
                return false;
            }
            const char* v = argv[++i];
            if (std::strcmp(v, "bfs") == 0)       opts.mode = false;
            else if (std::strcmp(v, "random") == 0) opts.mode = true;
            else {
                if (mpi_rank == 0) std::cerr << "오류: -m 인자는 bfs 또는 random 이어야 합니다: " << v << "\n";
                return false;
            }
            continue;
        }

        // --mode=bfs 같은 형식 허용
        if (std::strncmp(a, "--mode=", 7) == 0) {
            const char* v = a + 7;
            if (std::strcmp(v, "bfs") == 0)       opts.mode = false;
            else if (std::strcmp(v, "random") == 0) opts.mode = true;
            else {
                if (mpi_rank == 0) std::cerr << "오류: --mode 값은 bfs 또는 random 이어야 합니다: " << v << "\n";
                return false;
            }
            continue;
        }

        // 알 수 없는 -옵션
        if (a[0] == '-') {
            if (mpi_rank == 0) std::cerr << "오류: 알 수 없는 옵션: " << a << "\n";
            return false;
        }

        // 위치 인자
        positionals.push_back(a);
    }

    // 위치 인자: <graph_file> <num_partitions> <theta>
    if (positionals.size() != 3) {
        if (mpi_rank == 0) std::cerr << "오류: 위치 인자는 정확히 3개여야 합니다.\n";
        return false;
    }

    opts.graph_file     = positionals[0];
    opts.num_partitions = std::atoi(positionals[1]);
    opts.theta          = std::atoi(positionals[2]);

    if (opts.num_partitions <= 0 || opts.theta < 0) {
        if (mpi_rank == 0) std::cerr << "오류: num_partitions>0, theta>=0 이어야 합니다.\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    Options opts;
    if (!parse_args(argc, argv, opts, mpi_rank)) {
        if (mpi_rank == 0)
            std::cout << "Usage: mpirun -np <procs> ./hpc_partitioning [--verbose] -m <bfs|random> <graph_file> <num_partitions> <theta>\n";
        MPI_Finalize();
        return 1;
    }

    if (opts.verbose && mpi_rank == 0) {
        std::cout << "[옵션] mode=" << (opts.mode == false ? "bfs" : "random")
                  << ", verbose=on\n";
    }
    
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
            opts.graph_file,
            opts.num_partitions,
            opts.theta,
            opts.mode,
            opts.verbose,
            local_graph,
            ghost_nodes
    );

    
    PartitioningMetrics metrics1(metrics1_raw, opts.num_partitions);
    
    // Phase1 완료 동기화
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --------------------
    // GPU 할당 (Phase2 전에 미리 수행)
    // --------------------
    // if (mpi_rank == 0) {
    //     std::cout << "\n=== GPU 할당 ===\n";
    // }
    
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
        opts.num_partitions,
        local_graph,
        ghost_nodes,
        gpu_id
    );

    if (mpi_rank == 0) {
        metrics1_raw.print();
        printComparisonReport(metrics1, metrics2);
    }
    
    MPI_Finalize();
    return 0;
}