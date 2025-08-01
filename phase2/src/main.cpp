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

    if (argc < 3) {
        if (mpi_rank == 0)
            std::cout << "Usage: mpirun -np <servers> ./hpc_partitioning <graph_file> <num_partitions>\n";
        MPI_Finalize();
        return 1;
    }

    std::string graph_file = argv[1];
    int num_partitions = std::stoi(argv[2]);

    // Phase 1 데이터
    Graph local_graph;
    std::vector<int> vertex_labels;
    std::vector<int> global_ids;
    std::unordered_map<int, int> global_to_local;

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
        vertex_labels,     // 초기 라벨
        global_ids,        // 글로벌 ID 배열
        global_to_local);  // 글로벌→로컬 매핑

    PartitioningMetrics metrics1(metrics1_raw, num_partitions);
    // --------------------
    // Phase 2 실행
    // --------------------

    PartitioningMetrics metrics2 = run_phase2(
        mpi_rank, mpi_size,
        num_partitions,
        local_graph,
        vertex_labels,
        global_ids,
        global_to_local
    );

    if (mpi_rank == 0) {
        printComparisonReport(metrics1, metrics2);
    }
    MPI_Finalize();
    return 0;
}