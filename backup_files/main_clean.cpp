#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <mutex>
#include "types.h"
#include "phase1.h"
#include "mpi_workflow.h"  // 분리된 MPI 워크플로우 헤더

#define CUDA_CHECK(call) // CUDA 비활성화 시 빈 매크로

// 메인 함수
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc < 3) {
        if (mpi_rank == 0) {
            std::cout << "사용법: mpirun -np <서버수> ./mpi_distributed_workflow_v2 <그래프파일> <파티션수>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int num_partitions = std::atoi(argv[2]);
    std::string filename = argv[1];

    // Phase 1: 그래프 분할 및 분배 (새로운 파일에서)
    Graph local_graph;
    std::vector<int> vertex_labels;
    Phase1Metrics phase1_metrics = phase1_partition_and_distribute(mpi_rank, mpi_size, num_partitions, local_graph, vertex_labels, filename);

    // Phase 2: 7단계 알고리즘 (각 파티션별로 OpenMP 스레드 병렬)
    try {
        // 헤더파일에서 정의된 분리된 클래스 사용
        MPIDistributedWorkflowV2 workflow(argc, argv, local_graph, vertex_labels, phase1_metrics);
        workflow.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
