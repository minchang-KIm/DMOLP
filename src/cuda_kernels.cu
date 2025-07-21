#ifndef CUDA_KERNELS_CU
#define CUDA_KERNELS_CU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <omp.h>

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

// GPU 상수 메모리 (성능 최적화)
__constant__ PartitionInfoGPU d_partition_info[16]; // 최대 16개 파티션 지원

// CUDA 커널: Dynamic Label Propagation (통합 버전 - 모든 파티션 처리)
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
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_boundary_vertices) return;
    
    int vertex = boundary_vertices[tid];
    
    // 정점 ID 경계 체크
    if (vertex < 0 || vertex >= num_vertices) return;
    
    int current_label = vertex_labels[vertex];
    
    // 정점 소유권 확인: 이 MPI 프로세스가 소유한 정점만 라벨 변경 가능
    // 하지만 이웃 정점의 라벨은 Ghost Node로 복제된 정보를 참조 가능
    bool can_modify = (vertex >= start_vertex && vertex < end_vertex);
    
    // 공유 메모리: 라벨별 점수 계산
    extern __shared__ double label_scores[];
    
    // 스레드별 라벨 점수 초기화
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        label_scores[i] = 0.0;
    }
    __syncthreads();
    
    // 이웃들의 라벨별 점수 계산
    for (int edge_idx = row_ptr[vertex]; edge_idx < row_ptr[vertex + 1]; ++edge_idx) {
        int neighbor = col_indices[edge_idx];
        
        // 이웃 정점 경계 체크
        if (neighbor < 0 || neighbor >= num_vertices) continue;
        
        int neighbor_label = vertex_labels[neighbor];
        
        if (neighbor_label >= 0 && neighbor_label < num_partitions) {
            // Score(L) = |u| * (1 + P_L) 계산
            double score = 1.0 * (1.0 + d_partition_info[neighbor_label].P_L);
            atomicAdd(&label_scores[neighbor_label], score);
        }
    }
    __syncthreads();
    
    // 최고 점수 라벨 선택
    int best_label = current_label;
    double best_score = label_scores[current_label];
    
    for (int label = 0; label < num_partitions; ++label) {
        if (label_scores[label] > best_score) {
            best_score = label_scores[label];
            best_label = label;
        }
    }
    
    // 라벨 변경이 필요하고 소유권이 있는 경우에만 변경 (Ghost Node 방식)
    if (best_label != current_label && can_modify) {
        vertex_labels[vertex] = best_label;
        update_flags[vertex] = 1; // 업데이트 플래그 설정
        atomicAdd(label_changes, 1); // 전역 카운터 증가
    }
}

// CUDA 커널: Dynamic Label Propagation (파티션별 처리 - 문제 있는 버전)
__global__ void dynamicLabelPropagationKernel(
    int* vertex_labels,           // 정점 라벨 배열
    const int* row_ptr,          // CSR row pointer
    const int* col_indices,      // CSR column indices
    const int* boundary_vertices, // 경계 정점 배열
    int* label_changes,          // 라벨 변경 카운트 (atomic)
    int* update_flags,           // 업데이트 플래그 배열
    int num_boundary_vertices,   // 경계 정점 수
    int num_partitions,          // 파티션 수
    int thread_partition         // 이 커널이 처리할 파티션 ID
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_boundary_vertices) return;
    
    int vertex = boundary_vertices[tid];
    int current_label = vertex_labels[vertex];
    
    // 현재 파티션에 속한 정점만 처리
    if (current_label != thread_partition) return;
    
    // 공유 메모리: 라벨별 점수 계산
    extern __shared__ double label_scores[];
    
    // 스레드별 라벨 점수 초기화
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        label_scores[i] = 0.0;
    }
    __syncthreads();
    
    // 이웃들의 라벨별 점수 계산
    for (int edge_idx = row_ptr[vertex]; edge_idx < row_ptr[vertex + 1]; ++edge_idx) {
        int neighbor = col_indices[edge_idx];
        int neighbor_label = vertex_labels[neighbor];
        
        if (neighbor_label >= 0 && neighbor_label < num_partitions) {
            // Score(L) = |u| * (1 + P_L) 계산
            double score = 1.0 * (1.0 + d_partition_info[neighbor_label].P_L);
            atomicAdd(&label_scores[neighbor_label], score);
        }
    }
    __syncthreads();
    
    // 최고 점수 라벨 선택
    int best_label = current_label;
    double best_score = label_scores[current_label];
    
    for (int label = 0; label < num_partitions; ++label) {
        if (label_scores[label] > best_score) {
            best_score = label_scores[label];
            best_label = label;
        }
    }
    
    // 라벨 변경이 필요한 경우
    if (best_label != current_label) {
        vertex_labels[vertex] = best_label;
        update_flags[vertex] = 1; // 업데이트 플래그 설정
        atomicAdd(label_changes, 1); // 전역 카운터 증가
    }
}

// CUDA 커널: Boundary Vertex 추출
__global__ void extractBoundaryVerticesKernel(
    const int* vertex_labels,
    const int* row_ptr,
    const int* col_indices,
    int* boundary_flags,
    int num_vertices
) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vertex >= num_vertices) return;
    
    int vertex_label = vertex_labels[vertex];
    bool is_boundary = false;
    
    // 이웃들 확인
    for (int edge_idx = row_ptr[vertex]; edge_idx < row_ptr[vertex + 1]; ++edge_idx) {
        int neighbor = col_indices[edge_idx];
        if (neighbor < num_vertices) {
            int neighbor_label = vertex_labels[neighbor];
            if (vertex_label != neighbor_label) {
                is_boundary = true;
                break;
            }
        }
    }
    
    boundary_flags[vertex] = is_boundary ? 1 : 0;
}

// CUDA 커널: Edge-cut 계산
__global__ void calculateEdgeCutKernel(
    const int* vertex_labels,
    const int* row_ptr,
    const int* col_indices,
    int* edge_cut_count,
    int num_vertices
) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vertex >= num_vertices) return;
    
    int vertex_label = vertex_labels[vertex];
    int local_edge_cut = 0;
    
    // 이웃들과 라벨 비교 (중복 방지: vertex < neighbor)
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
        atomicAdd(edge_cut_count, local_edge_cut);
    }
}

// GPU 메모리 관리 클래스
class GPUMemoryManager {
private:
    // 디바이스 메모리 포인터들
    int* d_vertex_labels;
    int* d_row_ptr;
    int* d_col_indices;
    int* d_boundary_vertices;
    int* d_boundary_flags;
    int* d_label_changes;
    int* d_update_flags;
    int* d_edge_cut_count;
    
    size_t num_vertices;
    size_t num_edges;
    size_t max_boundary_vertices;
    
public:
    GPUMemoryManager(size_t vertices, size_t edges) 
        : num_vertices(vertices), num_edges(edges), max_boundary_vertices(vertices) {
        
        // GPU 메모리 할당
        CUDA_CHECK(cudaMalloc(&d_vertex_labels, num_vertices * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_indices, num_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_boundary_vertices, max_boundary_vertices * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_boundary_flags, num_vertices * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_label_changes, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_update_flags, num_vertices * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edge_cut_count, sizeof(int)));
        
        std::cout << "  [GPU] GPU 메모리 할당 완료: " << num_vertices << " 정점, " << num_edges << " 간선\n";
    }
    
    ~GPUMemoryManager() {
        // GPU 메모리 해제
        cudaFree(d_vertex_labels);
        cudaFree(d_row_ptr);
        cudaFree(d_col_indices);
        cudaFree(d_boundary_vertices);
        cudaFree(d_boundary_flags);
        cudaFree(d_label_changes);
        cudaFree(d_update_flags);
        cudaFree(d_edge_cut_count);
    }
    
    // 데이터 GPU로 복사
    void copyToGPU(const std::vector<int>& vertex_labels,
                   const std::vector<int>& row_ptr,
                   const std::vector<int>& col_indices) {
        CUDA_CHECK(cudaMemcpy(d_vertex_labels, vertex_labels.data(), 
                             num_vertices * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), 
                             (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_indices, col_indices.data(), 
                             num_edges * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // 데이터 CPU로 복사
    void copyToCPU(std::vector<int>& vertex_labels) {
        CUDA_CHECK(cudaMemcpy(vertex_labels.data(), d_vertex_labels, 
                             num_vertices * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    // Boundary Vertices 추출 (GPU)
    int extractBoundaryVertices(std::vector<int>& boundary_vertices) {
        // Step 1: Boundary flags 계산
        dim3 blockSize(256);
        dim3 gridSize((num_vertices + blockSize.x - 1) / blockSize.x);
        
        CUDA_CHECK(cudaMemset(d_boundary_flags, 0, num_vertices * sizeof(int)));
        
        extractBoundaryVerticesKernel<<<gridSize, blockSize>>>(
            d_vertex_labels, d_row_ptr, d_col_indices, d_boundary_flags, num_vertices
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 2: CPU에서 boundary vertices 수집 (OpenMP 활용)
        std::vector<int> boundary_flags(num_vertices);
        CUDA_CHECK(cudaMemcpy(boundary_flags.data(), d_boundary_flags, 
                             num_vertices * sizeof(int), cudaMemcpyDeviceToHost));
        
        boundary_vertices.clear();
        
        // OpenMP로 병렬 수집
        #pragma omp parallel
        {
            std::vector<int> local_boundary;
            
            #pragma omp for nowait
            for (int i = 0; i < num_vertices; ++i) {
                if (boundary_flags[i] == 1) {
                    local_boundary.push_back(i);
                }
            }
            
            #pragma omp critical
            {
                boundary_vertices.insert(boundary_vertices.end(), 
                                        local_boundary.begin(), local_boundary.end());
            }
        }
        
        return boundary_vertices.size();
    }
    
    // Edge-cut 계산 (GPU)
    int calculateEdgeCut() {
        dim3 blockSize(256);
        dim3 gridSize((num_vertices + blockSize.x - 1) / blockSize.x);
        
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_edge_cut_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
        
        calculateEdgeCutKernel<<<gridSize, blockSize>>>(
            d_vertex_labels, d_row_ptr, d_col_indices, d_edge_cut_count, num_vertices
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        int edge_cut;
        CUDA_CHECK(cudaMemcpy(&edge_cut, d_edge_cut_count, sizeof(int), cudaMemcpyDeviceToHost));
        
        return edge_cut;
    }
    
    // Dynamic Label Propagation 수행 (GPU + OpenMP 하이브리드)
    int performDynamicLabelPropagation(const std::vector<int>& boundary_vertices,
                                     const std::vector<PartitionInfoGPU>& partition_info,
                                     int num_partitions,
                                     int mpi_rank,
                                     int start_vertex,
                                     int end_vertex) {
        if (boundary_vertices.empty()) return 0;
        
        // 파티션 정보를 상수 메모리로 복사
        CUDA_CHECK(cudaMemcpyToSymbol(d_partition_info, partition_info.data(), 
                                     num_partitions * sizeof(PartitionInfoGPU)));
        
        // Boundary vertices를 GPU로 복사
        CUDA_CHECK(cudaMemcpy(d_boundary_vertices, boundary_vertices.data(), 
                             boundary_vertices.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        int total_updates = 0;
        
        // GPU 메모리 레이스 컨디션 방지를 위해 순차 실행
        // 전체 boundary vertices를 한 번에 처리 (파티션별 분할 제거)
        dim3 blockSize(256);
        dim3 gridSize((boundary_vertices.size() + blockSize.x - 1) / blockSize.x);
        size_t shared_mem_size = num_partitions * sizeof(double);
        
        // 라벨 변경 카운터 초기화
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_label_changes, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_update_flags, 0, num_vertices * sizeof(int)));
        
        // 통합 CUDA 커널 실행 (현재 MPI rank 파티션만 처리)
        dynamicLabelPropagationKernelUnified<<<gridSize, blockSize, shared_mem_size>>>(
            d_vertex_labels, d_row_ptr, d_col_indices, d_boundary_vertices,
            d_label_changes, d_update_flags, boundary_vertices.size(),
            num_partitions, mpi_rank, num_vertices, start_vertex, end_vertex
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 총 업데이트 수 가져오기
        CUDA_CHECK(cudaMemcpy(&total_updates, d_label_changes, 
                             sizeof(int), cudaMemcpyDeviceToHost));
        
        return total_updates;
    }
    
    // 메모리 포인터 접근자들
    int* getDeviceVertexLabels() { return d_vertex_labels; }
    int* getDeviceRowPtr() { return d_row_ptr; }
    int* getDeviceColIndices() { return d_col_indices; }
};

#endif // CUDA_KERNELS_CU
