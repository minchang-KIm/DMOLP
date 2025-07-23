#ifndef DMOLP_CUDA_KERNELS_H
#define DMOLP_CUDA_KERNELS_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <omp.h>
#include <memory>
#include <atomic>
#include <mutex>

// CUDA 에러 체크 매크로
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 파티션 정보 구조체 (GPU용)
struct PartitionInfoGPU {
    int partition_id;
    double RV;
    double RE;
    double P_L;
};

// CUDA 커널 함수 선언
__global__ void dynamicLabelPropagationKernelUnified(
    int* vertex_labels,           // 정점 라벨 배열
    const int* row_ptr,          // CSR row pointer
    const int* col_indices,      // CSR column indices
    const int* boundary_vertices, // 경계 정점 배열
    int* label_changes,          // 라벨 변경 카운트 (atomic)
    int* update_flags,           // 업데이트 플래그 배열
    int num_boundary_vertices,   // 경계 정점 수
    int num_partitions,          // 파티션 수
    int mpi_rank,                // 현재 MPI 프로세스 rank
    int num_vertices,            // 전체 정점 수 (경계 체크용)
    int start_vertex,            // 이 MPI 프로세스 소유 정점 시작 ID
    int end_vertex               // 이 MPI 프로세스 소유 정점 끝 ID (exclusive)
);

__global__ void calculateEdgeCutKernel(
    const int* vertex_labels,    // 정점 라벨 배열
    const int* row_ptr,          // CSR row pointer
    const int* col_indices,      // CSR column indices
    int* edge_cut,               // Edge-cut 결과 (atomic)
    int num_vertices             // 정점 수
);

// GPU 메모리 매니저 클래스
class GPUMemoryManager {
private:
    // GPU 메모리 포인터들
    int* d_vertex_labels_;
    int* d_row_ptr_;
    int* d_col_indices_;
    int* d_boundary_vertices_;
    int* d_label_changes_;
    int* d_update_flags_;
    int* d_edge_cut_;
    
    // 메모리 크기 정보
    size_t num_vertices_;
    size_t num_edges_;
    
    // GPU 컨텍스트 관리
    static std::atomic<int> next_gpu_;
    static std::vector<std::mutex> gpu_mutexes_;
    static int num_gpus_;

public:
    GPUMemoryManager(size_t num_vertices, size_t num_edges);
    ~GPUMemoryManager();
    
    // GPU 메모리 초기화 및 정리
    void allocateGPUMemory();
    void freeGPUMemory();
    
    // 데이터 복사
    void copyToGPU(const std::vector<int>& vertex_labels, 
                   const std::vector<int>& row_ptr, 
                   const std::vector<int>& col_indices);
    void copyToCPU(std::vector<int>& vertex_labels);
    
    // GPU 연산 수행
    int performDynamicLabelPropagation(const std::vector<int>& boundary_vertices,
                                      const std::vector<PartitionInfoGPU>& partition_info,
                                      int num_partitions, int mpi_rank,
                                      int start_vertex, int end_vertex);
    
    int calculateEdgeCut();
    
    // 유틸리티 함수
    void synchronize();
    static void initializeMultiGPU();
    static void finalizeMultiGPU();
};

#else
// CUDA 비활성화 시 빈 매크로 및 더미 클래스
#define CUDA_CHECK(call) // CUDA 비활성화 시 빈 매크로

// 더미 구조체 (컴파일 에러 방지)
struct PartitionInfoGPU {
    int partition_id;
    double RV;
    double RE;
    double P_L;
};

// 더미 클래스 (CUDA 비활성화 시)
class GPUMemoryManager {
public:
    GPUMemoryManager(size_t num_vertices, size_t num_edges) {}
    ~GPUMemoryManager() {}
    void copyToGPU(const std::vector<int>& vertex_labels, 
                   const std::vector<int>& row_ptr, 
                   const std::vector<int>& col_indices) {}
    void copyToCPU(std::vector<int>& vertex_labels) {}
    int performDynamicLabelPropagation(const std::vector<int>& boundary_vertices,
                                      const std::vector<PartitionInfoGPU>& partition_info,
                                      int num_partitions, int mpi_rank,
                                      int start_vertex, int end_vertex) { return 0; }
    int calculateEdgeCut() { return 0; }
    void synchronize() {}
    static void initializeMultiGPU() {}
    static void finalizeMultiGPU() {}
};

#endif // USE_CUDA

#endif // DMOLP_CUDA_KERNELS_H
