#pragma once
#include <vector>
#include "graph_types.h"  

// 고성능 GPU 라벨 전파 함수
void runBoundaryLPOnGPU_Optimized(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions);