#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>

#include "phase1/phase1.h"
#include "phase2/phase2.h"
#include "utils.hpp"

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