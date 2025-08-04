#include "phase1.h"
#include "phase2.h"
#include "report_utils.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc < 4) {
        if (mpi_rank == 0)
            std::cout << "Usage: mpirun -np <servers> ./hpc_partitioning <graph_file> <num_partitions> <thetta> \n";
        MPI_Finalize();
        return 1;
    }

    std::string graph_file = argv[1];
    int num_partitions = std::stoi(argv[2]);
    double thetta = std::stod(argv[3]);

    // Phase 1 데이터
    Graph local_graph;
    GhostNodes ghost_nodes;

    // --------------------
    // Phase 1 실행
    // --------------------
    if (mpi_rank == 0) {
        std::cout << "\n=== Phase 1: Initial Partitioning & Distribution ===\n";
    }

    Phase1Metrics metrics1_raw = phase1_partition_and_distribute(
        mpi_rank,
        mpi_size,
        num_partitions,
        graph_file,        // 파일 이름
        local_graph,       // 로컬 그래프
        ghost_nodes,       // Ghost 노드 정보
        thetta             // Thetta 값
    );

    PartitioningMetrics metrics1(metrics1_raw, num_partitions);
    // --------------------
    // Phase 2 실행
    // --------------------

    PartitioningMetrics metrics2 = run_phase2(
        mpi_rank, mpi_size,
        num_partitions,
        local_graph,
        ghost_nodes
    );

    if (mpi_rank == 0) {
        printComparisonReport(metrics1, metrics2);
    }
    MPI_Finalize();
    return 0;
}