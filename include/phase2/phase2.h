#pragma once
#include "graph_types.h"
#include <vector>
#include <unordered_map>
#include "utils.hpp"

PartitioningMetrics run_phase2(
    int mpi_rank, int mpi_size,
    int num_partitions,
    Graph& local_graph,
    GhostNodes& ghost_nodes
);