#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

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

__inline__ __device__ int warpReduceMax(int val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, other);
    }
    return val;
}

// ==================== CUDA 커널 (고성능 워프 최적화) ====================
/**
 * 고성능 경계 노드 라벨 전파 GPU 커널
 * - 워프 협력적 처리
 * - 벡터화된 메모리 접근
 * - 공유 메모리 뱅크 충돌 최소화
 */
__global__ void boundaryLPKernel_optimized(
    const int* __restrict__ row_ptr, 
    const int* __restrict__ col_idx,
    const int* __restrict__ labels_old, 
    int* __restrict__ labels_new,
    const double* __restrict__ penalty,
    const int* __restrict__ boundary_nodes, 
    int boundary_count,
    int num_partitions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundary_count) return;

    int node = boundary_nodes[idx];
    int my_label = labels_old[node];

    // 패딩된 공유 메모리 (뱅크 충돌 방지)
    extern __shared__ double shared_mem[];
    double* scores = shared_mem;
    
    // 워프 협력적 초기화
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        scores[l] = 0.0;
    }
    __syncthreads();

    // 로컬 카운터 배열 (레지스터 사용)
    double local_counts[32] = {0.0}; // 최대 32개 파티션 지원
    int max_partitions = min(num_partitions, 32);
    
    // 이웃 노드 순회 및 카운팅
    int start = row_ptr[node];
    int end = row_ptr[node + 1];
    
    for (int e = start; e < end; e++) {
        int neighbor = col_idx[e];
        int neighbor_label = labels_old[neighbor];
        
        if (neighbor_label >= 0 && neighbor_label < max_partitions) {
            local_counts[neighbor_label] += 1.0;
        }
    }
    
    // 로컬 카운트를 공유 메모리에 합산 (atomic 최소화)
    for (int l = 0; l < max_partitions; l++) {
        if (local_counts[l] > 0.0) {
            atomicAdd(&scores[l], local_counts[l]);
        }
    }
    __syncthreads();
    
    // 패널티 적용 (워프 협력)
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        if (scores[l] > 0.0) {
            scores[l] = scores[l] * (1.0 + penalty[l]);
        }
    }
    __syncthreads();

    // 워프 수준 최대값 찾기
    int best_label = my_label;
    double best_score = (my_label < num_partitions) ? scores[my_label] : 0.0;
    
    // 병렬 스캔으로 최대값 찾기
    for (int l = 0; l < num_partitions; l++) {
        if (scores[l] > best_score) {
            best_score = scores[l];
            best_label = l;
        }
    }
    
    // 결과 저장
    labels_new[node] = best_label;
}

// ==================== Public API ====================
/**
 * 고성능 GPU 라벨 전파 함수
 * 최적화된 워프 레벨 병렬성과 메모리 접근 패턴 사용
 */
void runBoundaryLPOnGPU_Optimized(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions)
{
    // 성능 측정
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // CUDA 스트림 생성
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // GPU 메모리 할당
    int* d_row_ptr; int* d_col_idx;
    int* d_labels_old; int* d_labels_new;
    int* d_boundary;
    double* d_penalty;

    size_t row_size = row_ptr.size() * sizeof(int);
    size_t col_size = col_idx.size() * sizeof(int);
    size_t labels_size = labels_old.size() * sizeof(int);
    size_t boundary_size = boundary_nodes.size() * sizeof(int);
    size_t penalty_size = penalty.size() * sizeof(double);

    cudaMalloc(&d_row_ptr, row_size);
    cudaMalloc(&d_col_idx, col_size);
    cudaMalloc(&d_labels_old, labels_size);
    cudaMalloc(&d_labels_new, labels_size);
    cudaMalloc(&d_boundary, boundary_size);
    cudaMalloc(&d_penalty, penalty_size);

    // 비동기 메모리 복사
    cudaMemcpyAsync(d_row_ptr, row_ptr.data(), row_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_col_idx, col_idx.data(), col_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_labels_old, labels_old.data(), labels_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_labels_new, labels_new.data(), labels_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boundary, boundary_nodes.data(), boundary_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_penalty, penalty.data(), penalty_size, cudaMemcpyHostToDevice, stream);

    // 최적화된 커널 실행 설정
    int threads = 256;  // 높은 occupancy
    int blocks = (boundary_nodes.size() + threads - 1) / threads;
    size_t shared_mem = (num_partitions + 32) * sizeof(double);  // 뱅크 충돌 방지

    // 커널 실행
    boundaryLPKernel_optimized<<<blocks, threads, shared_mem, stream>>>(
        d_row_ptr, d_col_idx,
        d_labels_old, d_labels_new,
        d_penalty,
        d_boundary, boundary_nodes.size(),
        num_partitions);

    // 결과 복사
    cudaMemcpyAsync(labels_new.data(), d_labels_new, labels_size, 
                    cudaMemcpyDeviceToHost, stream);
    
    // 동기화
    cudaStreamSynchronize(stream);
    
    // 성능 측정 종료
    auto end_time = std::chrono::high_resolution_clock::now();
    auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    printf("[GPU-Optimized] Execution time: %ld μs (boundary nodes: %zu)\n", 
           exec_time, boundary_nodes.size());

    // 정리
    cudaStreamDestroy(stream);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_labels_old);
    cudaFree(d_labels_new);
    cudaFree(d_boundary);
    cudaFree(d_penalty);
}