#pragma once
#include "graph_types.h"
#include <vector>
#include <unordered_map>
#include <string>



Phase1Metrics phase1_partition_and_distribute(
    int mpi_rank, int mpi_size, int num_partitions,
    const std::string& filename,
    Graph& local_graph,
    GhostNodes& ghost_nodes,
    double thetta
);
