#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

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

// ==================== 적응적 스케일링 함수 ====================
/**
 * 패널티 값들을 적응적으로 스케일링
 * 작은 패널티 차이를 증폭하여 실제 라벨 변경이 일어나도록 함
 */
static double calculateAdaptiveScaling(const std::vector<double>& penalties, bool enable_adaptive_scaling = true) {
    if (penalties.empty() || !enable_adaptive_scaling) return 1.0;
    
    double min_penalty = *std::min_element(penalties.begin(), penalties.end());
    double max_penalty = *std::max_element(penalties.begin(), penalties.end());
    double range = max_penalty - min_penalty;
    
    // 범위가 너무 작으면 스케일링 적용
    if (range < 0.1) {
        return 10.0;  // 10배 증폭
    } else if (range < 0.05) {
        return 20.0;  // 20배 증폭
    } else if (range < 0.01) {
        return 50.0;  // 50배 증폭
    }
    
    return 1.0;  // 스케일링 불필요
}

// ==================== CUDA 커널 (Atomic 방식) ====================
/**
 * 경계 노드들에 대해 라벨 전파를 수행하는 GPU 커널 (Atomic 연산 사용)
 * 각 경계 노드마다 이웃들의 라벨별 점수를 계산하여 최적 라벨 선택
 */
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

    // 공유 메모리에 각 라벨별 점수 저장
    extern __shared__ double scores[];
    
    // 점수 배열 초기화 (각 스레드가 담당하는 라벨들)
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        scores[l] = 0.0;
    }
    __syncthreads();

    // 현재 노드의 모든 이웃에 대해 점수 계산
    // Score(L) = |N_L| × (1 + P_L)
    for (int e = row_ptr[node]; e < row_ptr[node + 1]; e++) {
        int neighbor = col_idx[e];
        int neighbor_label = labels_old[neighbor];
        
        // 유효한 라벨인지 확인
        if (neighbor_label >= 0 && neighbor_label < num_partitions) {
            double score_contribution = 1.0 * (1.0 + PI[neighbor_label].P_L);
            atomicAdd(&scores[neighbor_label], score_contribution);
        }
    }
    __syncthreads();

    // 최고 점수를 가진 라벨 찾기
    int best_label = my_label;
    double best_score = scores[my_label];
    
    for (int l = 0; l < num_partitions; l++) {
        if (scores[l] > best_score) {
            best_score = scores[l];
            best_label = l;
        }
    }
    
    // 새로운 라벨 저장 (owned 노드만)
    labels_new[node] = best_label;
}

// ==================== CUDA 커널 (Warp 최적화 방식) ====================
/**
 * 경계 노드들에 대해 라벨 전파를 수행하는 GPU 커널 (Warp 최적화 사용)
 * penalty 배열을 직접 사용하여 더 간단하고 효율적인 구현
 */
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

    // 공유 메모리에 각 라벨별 점수 저장
    extern __shared__ double scores[];
    
    // 점수 배열 초기화 (각 스레드가 담당하는 라벨들)
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        scores[l] = 0.0;
    }
    __syncthreads();

    // 현재 노드의 모든 이웃에 대해 점수 계산
    // Score(L) = |N_L| × (1 + P_L)
    for (int e = row_ptr[node]; e < row_ptr[node + 1]; e++) {
        int neighbor = col_idx[e];  // owned 또는 ghost 노드
        int neighbor_label = labels_old[neighbor];
        
        // 유효한 라벨인지 확인
        if (neighbor_label >= 0 && neighbor_label < num_partitions) {
            double score_contribution = 1.0 * (1.0 + penalty[neighbor_label]);
            atomicAdd(&scores[neighbor_label], score_contribution);
        }
    }
    __syncthreads();

    // 최고 점수를 가진 라벨 찾기
    int best_label = my_label;
    double best_score = scores[my_label];
    
    for (int l = 0; l < num_partitions; l++) {
        if (scores[l] > best_score) {
            best_score = scores[l];
            best_label = l;
        }
    }
    
    // 새로운 라벨 저장 (owned 노드만)
    labels_new[node] = best_label;
}

// ==================== Public APIs ====================
/**
 * PartitionInfo를 사용하는 GPU 라벨 전파 함수 (Atomic 방식)
 * 적응적 스케일링을 적용하여 작은 패널티 차이도 효과적으로 활용
 */
void runBoundaryLPOnGPU(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<PartitionInfo>& PI,
    const std::vector<int>& boundary_nodes,
    int num_partitions,
    bool enable_adaptive_scaling)
{
    // 적응적 스케일링 계산
    std::vector<double> penalties(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        penalties[i] = PI[i].P_L;
    }
    double scaling_factor = calculateAdaptiveScaling(penalties, enable_adaptive_scaling);
    
    
    // PartitionInfo를 GPU용으로 변환 (스케일링 적용)
    std::vector<PartitionInfoGPU> PI_gpu(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        PI_gpu[i].P_L = PI[i].P_L * scaling_factor;
    }
    
    // GPU 메모리 할당
    PartitionInfoGPU* d_PI;
    int* d_row_ptr; int* d_col_idx;
    int* d_labels_old; int* d_labels_new;
    int* d_boundary;

    cudaMalloc(&d_PI, PI_gpu.size() * sizeof(PartitionInfoGPU));
    cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(int));
    cudaMalloc(&d_col_idx, col_idx.size() * sizeof(int));
    cudaMalloc(&d_labels_old, labels_old.size() * sizeof(int));
    cudaMalloc(&d_labels_new, labels_new.size() * sizeof(int));
    cudaMalloc(&d_boundary, boundary_nodes.size() * sizeof(int));

    // 데이터 복사
    cudaMemcpy(d_PI, PI_gpu.data(), PI_gpu.size() * sizeof(PartitionInfoGPU), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels_old, labels_old.data(), labels_old.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels_new, labels_new.data(), labels_new.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundary, boundary_nodes.data(), boundary_nodes.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 커널 실행 설정
    int threads = 128;
    int blocks = (boundary_nodes.size() + threads - 1) / threads;
    size_t shared_mem = num_partitions * sizeof(double);

    // 커널 실행
    boundaryLPKernel_atomic<<<blocks, threads, shared_mem>>>(
        d_row_ptr, d_col_idx,
        d_labels_old, d_labels_new,
        d_PI,
        d_boundary, boundary_nodes.size(),
        num_partitions);
    
    // 동기화 및 결과 복사
    cudaDeviceSynchronize();
    cudaMemcpy(labels_new.data(), d_labels_new, labels_new.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_PI);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_labels_old);
    cudaFree(d_labels_new);
    cudaFree(d_boundary);
}

/**
 * penalty 배열을 사용하는 GPU 라벨 전파 함수 (Warp 방식)
 * 적응적 스케일링을 적용하여 작은 패널티 차이도 효과적으로 활용
 */
void runBoundaryLPOnGPU_Warp(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions,
    bool enable_adaptive_scaling)
{
    // 적응적 스케일링 적용
    double scaling_factor = calculateAdaptiveScaling(penalty, enable_adaptive_scaling);
    std::vector<double> scaled_penalty(penalty.size());
    for (size_t i = 0; i < penalty.size(); i++) {
        scaled_penalty[i] = penalty[i] * scaling_factor;
    }
    
    // GPU 메모리 할당
    int* d_row_ptr; int* d_col_idx;
    int* d_labels_old; int* d_labels_new;
    int* d_boundary;
    double* d_penalty;

    cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(int));
    cudaMalloc(&d_col_idx, col_idx.size() * sizeof(int));
    cudaMalloc(&d_labels_old, labels_old.size() * sizeof(int));
    cudaMalloc(&d_labels_new, labels_new.size() * sizeof(int));
    cudaMalloc(&d_boundary, boundary_nodes.size() * sizeof(int));
    cudaMalloc(&d_penalty, scaled_penalty.size() * sizeof(double));

    // 데이터 복사 (스케일링된 패널티 사용)
    cudaMemcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels_old, labels_old.data(), labels_old.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels_new, labels_new.data(), labels_new.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundary, boundary_nodes.data(), boundary_nodes.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_penalty, scaled_penalty.data(), scaled_penalty.size() * sizeof(double), cudaMemcpyHostToDevice);

    // 커널 실행 설정
    int threads = 128;
    int blocks = (boundary_nodes.size() + threads - 1) / threads;
    size_t shared_mem = num_partitions * sizeof(double);

    // 커널 실행
    boundaryLPKernel_warp<<<blocks, threads, shared_mem>>>(
        d_row_ptr, d_col_idx,
        d_labels_old, d_labels_new,
        d_penalty,
        d_boundary, boundary_nodes.size(),
        num_partitions);
    
    // 동기화 및 결과 복사
    cudaDeviceSynchronize();
    cudaMemcpy(labels_new.data(), d_labels_new, labels_new.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_labels_old);
    cudaFree(d_labels_new);
    cudaFree(d_boundary);
    cudaFree(d_penalty);
}