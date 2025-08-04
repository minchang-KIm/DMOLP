#pragma once
#include "graph_types.h"
#include "phase1.h"
#include <vector>
#include <unordered_map>
#include "report_utils.h"

PartitioningMetrics run_phase2(
    int mpi_rank, int mpi_size,
    int num_partitions,
    Graph& local_graph,
    GhostNodes& ghost_nodes
);