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
    std::vector<int>& vertex_labels,
    const std::vector<int>& global_ids,
    const std::unordered_map<int,int>& global_to_local
);

// void calculatePartitionRatios(
//     const Graph &g,
//     const std::vector<int> &labels,
//     int num_partitions,
//     std::vector<PartitionInfo> &PI
// );