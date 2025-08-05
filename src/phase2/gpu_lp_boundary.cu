#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include "phase2/gpu_lp_boundary.h"

// GPU용 Partition Info 구조체
struct PartitionInfoGPU {
    double P_L;   // DMOLP Penalty
};

// ==================== Warp Reduction Helper ====================
__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ==================== CUDA 커널 (Shared memory atomic) ====================
__global__ void boundaryLPKernel_atomic(
    const int* row_ptr, const int* col_idx,
    const int* labels_old, int* labels_new,
    const PartitionInfoGPU* PI,
    const int* boundary_nodes, int boundary_count,
    int num_partitions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundary_count) return;

    int node = boundary_nodes[idx];
    int my_label = labels_old[node];

    extern __shared__ double score[];
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x)
        score[i] = 0.0;
    __syncthreads();

    // 이웃 라벨별 점수 계산
    for (int e = row_ptr[node]; e < row_ptr[node + 1]; e++) {
        int n = col_idx[e];
        int lbl = labels_old[n];
        atomicAdd(&score[lbl], 1.0 * (1.0 + PI[lbl].P_L));
    }
    __syncthreads();

    int best_label = my_label;
    double best_score = score[my_label];
    for (int l = 0; l < num_partitions; l++) {
        double val = score[l];
        if (val > best_score) {
            best_score = val;
            best_label = l;
        }
    }
    labels_new[node] = best_label;
}

// ==================== CUDA 커널 (Warp Reduction) ====================
__global__ void boundaryLPKernel_warp(
    const int* row_ptr, const int* col_idx,
    const int* labels_old, int* labels_new,
    const double* penalty,
    const int* boundary_nodes, int boundary_count,
    int num_partitions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundary_count) return;

    int node = boundary_nodes[idx];
    int my_label = labels_old[node];

    // 라벨별 점수를 warp 방식으로 계산
    extern __shared__ double s_scores[];
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x)
        s_scores[l] = 0.0;
    __syncthreads();

    // 간선 순회
    for (int e = row_ptr[node]; e < row_ptr[node + 1]; e++) {
        int n = col_idx[e];
        int lbl = labels_old[n];
        double score = 1.0 * (1.0 + penalty[lbl]);
        atomicAdd(&s_scores[lbl], score);
    }
    __syncthreads();

    int best_label = my_label;
    double best_score = s_scores[my_label];
    for (int l = 0; l < num_partitions; l++) {
        double val = s_scores[l];
        if (val > best_score) {
            best_score = val;
            best_label = l;
        }
    }
    labels_new[node] = best_label;
}

// ==================== 공통 GPU 실행 코드 ====================
static void gpu_launch_common(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const double* d_penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions,
    bool use_warp)
{
    int* d_row_ptr; int* d_col_idx;
    int* d_labels_old; int* d_labels_new;
    int* d_boundary;

    cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(int));
    cudaMalloc(&d_col_idx, col_idx.size() * sizeof(int));
    cudaMalloc(&d_labels_old, labels_old.size() * sizeof(int));
    cudaMalloc(&d_labels_new, labels_new.size() * sizeof(int));
    cudaMalloc(&d_boundary, boundary_nodes.size() * sizeof(int));

    cudaMemcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels_old, labels_old.data(), labels_old.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels_new, labels_new.data(), labels_new.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundary, boundary_nodes.data(), boundary_nodes.size() * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (boundary_nodes.size() + threads - 1) / threads;
    size_t shared_mem = num_partitions * sizeof(double);

    if (use_warp) {
        boundaryLPKernel_warp<<<blocks, threads, shared_mem>>>(
            d_row_ptr, d_col_idx,
            d_labels_old, d_labels_new,
            d_penalty,
            d_boundary, boundary_nodes.size(),
            num_partitions);
    } else {
        // 기존 방식
        PartitionInfoGPU* d_PI;
        std::vector<PartitionInfoGPU> PI_gpu(num_partitions);
        for (int i = 0; i < num_partitions; i++) PI_gpu[i].P_L = 0.0; // placeholder
        cudaMalloc(&d_PI, PI_gpu.size() * sizeof(PartitionInfoGPU));
        cudaMemcpy(d_PI, PI_gpu.data(), PI_gpu.size() * sizeof(PartitionInfoGPU), cudaMemcpyHostToDevice);

        boundaryLPKernel_atomic<<<blocks, threads, shared_mem>>>(
            d_row_ptr, d_col_idx,
            d_labels_old, d_labels_new,
            d_PI,
            d_boundary, boundary_nodes.size(),
            num_partitions);
        cudaFree(d_PI);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(labels_new.data(), d_labels_new, labels_new.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_labels_old);
    cudaFree(d_labels_new);
    cudaFree(d_boundary);
}

// ==================== Public APIs ====================
void runBoundaryLPOnGPU(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<PartitionInfo>& PI,
    const std::vector<int>& boundary_nodes,
    int num_partitions)
{
    gpu_launch_common(row_ptr, col_idx, labels_old, labels_new, nullptr, boundary_nodes, num_partitions, false);
}

void runBoundaryLPOnGPU_Warp(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions)
{
    double* d_penalty;
    cudaMalloc(&d_penalty, penalty.size() * sizeof(double));
    cudaMemcpy(d_penalty, penalty.data(), penalty.size() * sizeof(double), cudaMemcpyHostToDevice);

    gpu_launch_common(row_ptr, col_idx, labels_old, labels_new, d_penalty, boundary_nodes, num_partitions, true);

    cudaFree(d_penalty);
}