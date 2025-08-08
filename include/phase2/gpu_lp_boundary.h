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

// Thrust 라이브러리 기반 초고성능 GPU 라벨 전파 함수
void runBoundaryLPOnGPU_Thrust_Optimized(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions);

// 안전한 GPU 라벨 전파 함수 (fallback)
void runBoundaryLPOnGPU_Safe(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions);

// Pinned Memory 최적화 GPU 라벨 전파 함수 (메모리 누수 방지 + 성능 최적화)
void runBoundaryLPOnGPU_PinnedOptimized(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions);

// GPU 리소스 정리 함수
void cleanupGPUResources();