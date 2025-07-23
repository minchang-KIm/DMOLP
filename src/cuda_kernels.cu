#ifdef USE_CUDA

#include "cuda_kernels.h"
#include <iostream>

// 간단한 CUDA 커널들
__global__ void dynamicLabelPropagationKernelUnified(
    int* vertex_labels,
    const int* row_ptr,
    const int* col_indices,
    const int* boundary_vertices,
    int* label_changes,
    int* update_flags,
    int num_boundary_vertices,
    int num_partitions,
    int mpi_rank,
    int num_vertices,
    int start_vertex,
    int end_vertex
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_boundary_vertices) return;
    
    int vertex = boundary_vertices[tid];
    if (vertex < start_vertex || vertex >= end_vertex) return;
    
    int current_label = vertex_labels[vertex];
    int best_label = current_label;
    double best_score = 0.0;
    
    // 이웃 탐색
    for (int edge_idx = row_ptr[vertex]; edge_idx < row_ptr[vertex + 1]; ++edge_idx) {
        int neighbor = col_indices[edge_idx];
        int neighbor_label = vertex_labels[neighbor];
        
        if (neighbor_label != current_label && neighbor_label < num_partitions) {
            double score = 1.0;
            if (score > best_score) {
                best_score = score;
                best_label = neighbor_label;
            }
        }
    }
    
    if (best_label != current_label) {
        vertex_labels[vertex] = best_label;
        atomicAdd(label_changes, 1);
    }
}

__global__ void calculateEdgeCutKernel(
    const int* vertex_labels,
    const int* row_ptr,
    const int* col_indices,
    int* edge_cut,
    int num_vertices
) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= num_vertices) return;
    
    int vertex_label = vertex_labels[vertex];
    int local_edge_cut = 0;
    
    for (int edge_idx = row_ptr[vertex]; edge_idx < row_ptr[vertex + 1]; ++edge_idx) {
        int neighbor = col_indices[edge_idx];
        if (neighbor < num_vertices && vertex < neighbor) {
            int neighbor_label = vertex_labels[neighbor];
            if (vertex_label != neighbor_label) {
                local_edge_cut++;
            }
        }
    }
    
    if (local_edge_cut > 0) {
        atomicAdd(edge_cut, local_edge_cut);
    }
}

// GPUMemoryManager 구현
GPUMemoryManager::GPUMemoryManager(size_t num_vertices, size_t num_edges) 
    : num_vertices_(num_vertices), num_edges_(num_edges) {
    
    CUDA_CHECK(cudaMalloc(&d_vertex_labels_, num_vertices_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr_, (num_vertices_ + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_indices_, num_edges_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_boundary_vertices_, num_vertices_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_label_changes_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_update_flags_, num_vertices_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_cut_, sizeof(int)));
    
    std::cout << "  [GPU] GPU 메모리 할당 완료: " << num_vertices_ << " 정점, " << num_edges_ << " 간선\n";
}

GPUMemoryManager::~GPUMemoryManager() {
    cudaFree(d_vertex_labels_);
    cudaFree(d_row_ptr_);
    cudaFree(d_col_indices_);
    cudaFree(d_boundary_vertices_);
    cudaFree(d_label_changes_);
    cudaFree(d_update_flags_);
    cudaFree(d_edge_cut_);
}

void GPUMemoryManager::copyToGPU(const std::vector<int>& vertex_labels,
                                 const std::vector<int>& row_ptr,
                                 const std::vector<int>& col_indices) {
    CUDA_CHECK(cudaMemcpy(d_vertex_labels_, vertex_labels.data(), 
                         num_vertices_ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr_, row_ptr.data(), 
                         (num_vertices_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices_, col_indices.data(), 
                         num_edges_ * sizeof(int), cudaMemcpyHostToDevice));
}

void GPUMemoryManager::copyToCPU(std::vector<int>& vertex_labels) {
    CUDA_CHECK(cudaMemcpy(vertex_labels.data(), d_vertex_labels_, 
                         num_vertices_ * sizeof(int), cudaMemcpyDeviceToHost));
}

int GPUMemoryManager::calculateEdgeCut() {
    dim3 blockSize(256);
    dim3 gridSize((num_vertices_ + blockSize.x - 1) / blockSize.x);
    
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_edge_cut_, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    calculateEdgeCutKernel<<<gridSize, blockSize>>>(
        d_vertex_labels_, d_row_ptr_, d_col_indices_, d_edge_cut_, num_vertices_
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int edge_cut;
    CUDA_CHECK(cudaMemcpy(&edge_cut, d_edge_cut_, sizeof(int), cudaMemcpyDeviceToHost));
    
    return edge_cut;
}

int GPUMemoryManager::performDynamicLabelPropagation(const std::vector<int>& boundary_vertices,
                                                     const std::vector<PartitionInfoGPU>& partition_info,
                                                     int num_partitions,
                                                     int mpi_rank,
                                                     int start_vertex,
                                                     int end_vertex) {
    if (boundary_vertices.empty()) return 0;
    
    CUDA_CHECK(cudaMemcpy(d_boundary_vertices_, boundary_vertices.data(), 
                         boundary_vertices.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 blockSize(256);
    dim3 gridSize((boundary_vertices.size() + blockSize.x - 1) / blockSize.x);
    
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_label_changes_, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    dynamicLabelPropagationKernelUnified<<<gridSize, blockSize>>>(
        d_vertex_labels_, d_row_ptr_, d_col_indices_, d_boundary_vertices_,
        d_label_changes_, d_update_flags_, boundary_vertices.size(),
        num_partitions, mpi_rank, num_vertices_, start_vertex, end_vertex
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int total_updates;
    CUDA_CHECK(cudaMemcpy(&total_updates, d_label_changes_, sizeof(int), cudaMemcpyDeviceToHost));
    
    return total_updates;
}

void GPUMemoryManager::synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUMemoryManager::allocateGPUMemory() {
    // 이미 생성자에서 할당됨
}

void GPUMemoryManager::freeGPUMemory() {
    // 이미 소멸자에서 해제됨
}

void GPUMemoryManager::initializeMultiGPU() {
    // TODO: 멀티 GPU 초기화
}

void GPUMemoryManager::finalizeMultiGPU() {
    // TODO: 멀티 GPU 정리
}

#endif // USE_CUDA
