#pragma once
#include <vector>
#include "graph_types.h"  

void runBoundaryLPOnGPU(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<PartitionInfo>& PI,
    const std::vector<int>& boundary_nodes,
    int num_partitions,
    bool enable_adaptive_scaling = true);

void runBoundaryLPOnGPU_Warp(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions,
    bool enable_adaptive_scaling = true);