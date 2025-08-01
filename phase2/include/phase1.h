#pragma once
#include "graph_types.h"
#include <vector>
#include <unordered_map>
#include <string>



Phase1Metrics phase1_partition_and_distribute(
    int mpi_rank, int mpi_size, int num_partitions,
    const std::string& filename,
    Graph& local_graph,
    std::vector<int>& vertex_labels,
    std::vector<int>& global_ids,
    std::unordered_map<int,int>& global_to_local
);
