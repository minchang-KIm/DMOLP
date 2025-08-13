#include <cuda_runtime.h>
#ifdef function
#undef function
#endif
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <chrono>

#include "phase2/gpu_lp_boundary.h"

// ==================== 바운더리 서브그래프 생성 (로컬+고스트 통합) ====================

/**
 * 바운더리 노드 + 1-hop 이웃으로 구성된 통합 서브그래프 생성 (최적화)
 * - 로컬 노드와 고스트 노드의 라벨을 통합하여 GPU에 전달
 * - 로컬 노드만 업데이트 대상으로 표시
 * - GPU 메모리 지역성 최적화 + OpenMP 병렬화
 */
BoundarySubgraph createBoundarySubgraphUnified(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& boundary_nodes,
    const std::vector<int>& local_labels,
    const std::vector<int>& ghost_labels,
    const std::vector<int>& global_ids,
    int num_local_nodes)
{
    BoundarySubgraph subgraph;
    subgraph.num_local_nodes = num_local_nodes;
    int total_nodes = local_labels.size() + ghost_labels.size();
    
    // 1단계: 서브그래프에 포함될 모든 노드 수집 (바운더리 + 1-hop 이웃) - 병렬화
    std::vector<bool> node_included(total_nodes, false);
    
    // 바운더리 노드들 먼저 마킹
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < boundary_nodes.size(); i++) {
        int boundary_node = boundary_nodes[i];
        if (boundary_node >= 0 && boundary_node < total_nodes) {
            node_included[boundary_node] = true;
        }
    }
    
    // 각 바운더리 노드의 1-hop 이웃 추가 - 병렬화
    #pragma omp parallel for schedule(dynamic, 10)
    for (size_t i = 0; i < boundary_nodes.size(); i++) {
        int boundary_node = boundary_nodes[i];
        if (boundary_node >= 0 && boundary_node < (int)row_ptr.size() - 1) {
            for (int edge_idx = row_ptr[boundary_node]; edge_idx < row_ptr[boundary_node + 1]; edge_idx++) {
                int neighbor = col_idx[edge_idx];
                if (neighbor >= 0 && neighbor < total_nodes) {
                    node_included[neighbor] = true;
                }
            }
        }
    }
    
    // 2단계: 포함된 노드들을 벡터로 변환 (병렬 압축)
    std::vector<int> subgraph_nodes;
    
    // 더 정확한 예약: 바운더리 노드 수 + 평균 degree 추정
    size_t estimated_size = boundary_nodes.size() * 2; // 바운더리 + 1-hop 이웃 추정
    subgraph_nodes.reserve(std::min(estimated_size, static_cast<size_t>(total_nodes)));
    
    for (int i = 0; i < total_nodes; i++) {
        if (node_included[i]) {
            subgraph_nodes.push_back(i);
        }
    }
    
    subgraph.num_nodes = subgraph_nodes.size();
    subgraph.node_mapping = std::move(subgraph_nodes);
    subgraph.reverse_mapping.assign(total_nodes, -1);
    
    // 통합 라벨 배열 및 로컬 노드 플래그 구성 - 병렬화
    subgraph.labels.resize(subgraph.num_nodes);
    subgraph.local_node_flags.resize(subgraph.num_nodes);
    
    #pragma omp parallel for
    for (int i = 0; i < subgraph.num_nodes; i++) {
        int orig_node = subgraph.node_mapping[i];
        subgraph.reverse_mapping[orig_node] = i;
        
        // 라벨 설정 (로컬 또는 고스트)
        if (orig_node < num_local_nodes) {
            // 로컬 노드
            subgraph.labels[i] = local_labels[orig_node];
            subgraph.local_node_flags[i] = 1;
        } else {
            // 고스트 노드
            int ghost_idx = orig_node - num_local_nodes;
            if (ghost_idx >= 0 && ghost_idx < (int)ghost_labels.size()) {
                subgraph.labels[i] = ghost_labels[ghost_idx];
            } else {
                subgraph.labels[i] = -1; // 유효하지 않은 라벨
            }
            subgraph.local_node_flags[i] = 0;
        }
    }
    
    // 3단계: 서브그래프 CSR 구성 - 메모리 효율적 방식
    subgraph.row_ptr.resize(subgraph.num_nodes + 1, 0);
    
    // 먼저 각 노드의 이웃 수 계산 (병렬 + 캐시 친화적)
    std::vector<int> neighbor_counts(subgraph.num_nodes, 0);
    
    #pragma omp parallel for schedule(static, 64) // 캐시 라인 크기 고려
    for (int i = 0; i < subgraph.num_nodes; i++) {
        int orig_node = subgraph.node_mapping[i];
        if (orig_node < (int)row_ptr.size() - 1) {
            for (int edge_idx = row_ptr[orig_node]; edge_idx < row_ptr[orig_node + 1]; edge_idx++) {
                int neighbor = col_idx[edge_idx];
                if (neighbor >= 0 && neighbor < total_nodes && subgraph.reverse_mapping[neighbor] != -1) {
                    neighbor_counts[i]++;
                }
            }
        }
    }
    
    // CSR row_ptr 계산 (prefix sum)
    int edge_count = 0;
    for (int i = 0; i < subgraph.num_nodes; i++) {
        subgraph.row_ptr[i] = edge_count;
        edge_count += neighbor_counts[i];
    }
    subgraph.row_ptr[subgraph.num_nodes] = edge_count;
    subgraph.num_edges = edge_count;
    
    // 이웃 노드 수집 (병렬 + 메모리 지역성 최적화)
    subgraph.col_idx.resize(edge_count);
    
    #pragma omp parallel for schedule(static, 32)
    for (int i = 0; i < subgraph.num_nodes; i++) {
        int orig_node = subgraph.node_mapping[i];
        int start_idx = subgraph.row_ptr[i];
        int idx = start_idx;
        
        if (orig_node < (int)row_ptr.size() - 1) {
            for (int edge_idx = row_ptr[orig_node]; edge_idx < row_ptr[orig_node + 1]; edge_idx++) {
                int neighbor = col_idx[edge_idx];
                if (neighbor >= 0 && neighbor < total_nodes) {
                    int neighbor_subgraph_idx = subgraph.reverse_mapping[neighbor];
                    if (neighbor_subgraph_idx != -1) {
                        subgraph.col_idx[idx++] = neighbor_subgraph_idx;
                    }
                }
            }
        }
    }
    
    // 4단계: 서브그래프 내 실제 바운더리 노드 인덱스 찾기 (로컬 노드만) - 최적화
    std::vector<bool> is_boundary(boundary_nodes.size(), true);
    std::vector<int> boundary_subgraph_indices;
    boundary_subgraph_indices.reserve(boundary_nodes.size());
    
    for (int boundary_node : boundary_nodes) {
        if (boundary_node < num_local_nodes) { // 로컬 노드만
            int subgraph_idx = subgraph.reverse_mapping[boundary_node];
            if (subgraph_idx != -1) {
                boundary_subgraph_indices.push_back(subgraph_idx);
            }
        }
    }
    
    subgraph.boundary_indices = std::move(boundary_subgraph_indices);
    
    return subgraph;
}

/**
 * 적응적 바운더리 확장: 이전 바운더리 + 1-hop 이웃에서 실제 바운더리만 필터링
 */
std::vector<int> expandBoundaryNodes(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& prev_boundary_nodes,
    const std::vector<int>& labels,
    int labels_count)
{
    std::unordered_set<int> candidate_nodes;
    
    // 이전 바운더리 노드들과 그들의 1-hop 이웃 수집
    for (int boundary_node : prev_boundary_nodes) {
        if (boundary_node >= 0 && boundary_node < labels_count) {
            candidate_nodes.insert(boundary_node);
            
            if (boundary_node < (int)row_ptr.size() - 1) {
                for (int edge_idx = row_ptr[boundary_node]; edge_idx < row_ptr[boundary_node + 1]; edge_idx++) {
                    int neighbor = col_idx[edge_idx];
                    if (neighbor >= 0 && neighbor < labels_count) {
                        candidate_nodes.insert(neighbor);
                    }
                }
            }
        }
    }
    
    // 후보 노드들 중에서 실제 바운더리 노드만 필터링
    std::vector<int> new_boundary_nodes;
    for (int node : candidate_nodes) {
        if (node >= 0 && node < labels_count && node < (int)row_ptr.size() - 1) {
            int node_label = labels[node];
            bool is_boundary = false;
            
            for (int edge_idx = row_ptr[node]; edge_idx < row_ptr[node + 1]; edge_idx++) {
                int neighbor = col_idx[edge_idx];
                if (neighbor >= 0 && neighbor < labels_count) {
                    int neighbor_label = labels[neighbor];
                    if (neighbor_label != node_label) {
                        is_boundary = true;
                        break;
                    }
                }
            }
            
            if (is_boundary) {
                new_boundary_nodes.push_back(node);
            }
        }
    }
    
    return new_boundary_nodes;
}

// ==================== 단순화된 GPU 메모리 관리 ====================
class GPUMemoryManager {
public:
    static cudaError_t safeMalloc(void** ptr, size_t size) {
        return cudaMalloc(ptr, size);
    }
    
    static cudaError_t safeFree(void* ptr) {
        return cudaFree(ptr);
    }
};

// ==================== 서브그래프 전용 커널 (로컬 노드만 업데이트) ====================

/**
 * 통합 서브그래프 전용 최적화 커널 (워프 기반 처리)
 * - 각 워프(32개 스레드)가 하나의 바운더리 노드를 협력 처리
 * - 워프 내 스레드들이 이웃 노드들을 병렬로 처리
 * - 큰 degree 노드에서 성능 향상
 */
__global__ void boundaryLPKernel_unified(
    const int* __restrict__ row_ptr, 
    const int* __restrict__ col_idx,
    const int* __restrict__ labels_old, 
    int* __restrict__ labels_new,
    const int* __restrict__ local_node_flags,
    const double* __restrict__ penalty,
    const int* __restrict__ boundary_indices, 
    int boundary_count,
    int num_partitions,
    int subgraph_size)
{
    // 워프 관련 정보
    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_in_block = threadIdx.x / WARP_SIZE;
    
    // 워프가 담당할 바운더리 노드 결정
    if (warp_id >= boundary_count) return;
    
    int subgraph_node_idx = boundary_indices[warp_id];
    if (subgraph_node_idx < 0 || subgraph_node_idx >= subgraph_size) return;
    
    // 로컬 노드가 아니면 조기 종료 (고스트 노드는 업데이트하지 않음)
    if (local_node_flags[subgraph_node_idx] != 1) return;
    
    int my_label = labels_old[subgraph_node_idx];
    
    // 워프별 공유 메모리: 각 워프가 독립적인 스코어 배열 사용
    // 블록당 최대 8개 워프 (256/32), 최대 32개 파티션
    __shared__ double warp_scores[8][32];
    
    int effective_partitions = min(num_partitions, 32);
    
    // 스코어 배열 초기화: 각 스레드가 담당 파티션을 0으로 설정
    if (lane_id < effective_partitions) {
        warp_scores[warp_in_block][lane_id] = 0.0;
    }
    __syncwarp(); // 워프 내 모든 스레드 동기화
    
    // 이웃 노드 처리: stride 방식으로 병렬 순회
    int start = row_ptr[subgraph_node_idx];
    int end = row_ptr[subgraph_node_idx + 1];
    
    // 각 스레드가 stride=32로 이웃들을 처리
    // Thread 0: edge 0, 32, 64, ...
    // Thread 1: edge 1, 33, 65, ...
    for (int e = start + lane_id; e < end; e += WARP_SIZE) {
        int neighbor_idx = col_idx[e];
        if (neighbor_idx >= 0 && neighbor_idx < subgraph_size) {
            int neighbor_label = labels_old[neighbor_idx];
            if (neighbor_label >= 0 && neighbor_label < effective_partitions) {
                // atomic 연산으로 스레드 간 경쟁 상태 방지
                atomicAdd(&warp_scores[warp_in_block][neighbor_label], 1.0);
            }
        }
    }
    __syncwarp(); // 모든 이웃 처리 완료까지 대기
    
    // 워프 대표 스레드(lane 0)가 최종 라벨 결정
    if (lane_id == 0) {
        // 1단계: 패널티 적용하여 최종 스코어 계산
        for (int l = 0; l < effective_partitions; l++) {
            if (warp_scores[warp_in_block][l] > 0.0) {
                warp_scores[warp_in_block][l] = warp_scores[warp_in_block][l] * (1.0 + penalty[l]);
            }
        }
        
        // 2단계: 최고 스코어를 가진 라벨 찾기
        int best_label = my_label;
        double best_score = (my_label >= 0 && my_label < effective_partitions) ? 
                           warp_scores[warp_in_block][my_label] : 0.0;
        
        for (int l = 0; l < effective_partitions; l++) {
            if (warp_scores[warp_in_block][l] > best_score) {
                best_score = warp_scores[warp_in_block][l];
                best_label = l;
            }
        }
        
        // 3단계: 새 라벨 저장
        labels_new[subgraph_node_idx] = best_label;
    }
}

// ==================== GPU 처리 함수 ====================

/**
 * GPU 기반 바운더리 라벨 전파 - 통합 서브그래프 처리
 * 
 * 특징:
 * - 바운더리 서브그래프 기반으로 메모리 효율성 극대화
 * - 워프 기반 병렬 처리로 GPU 성능 최적화
 * - 로컬 노드만 업데이트하여 MPI 일관성 보장
 * - 스트리밍 방식으로 대용량 그래프 처리 지원
 */
GPULabelUpdateResult runBoundaryLPOnGPU_SubgraphUnified(
    const BoundarySubgraph& subgraph,
    const std::vector<double>& penalty,
    int num_partitions)
{
    GPULabelUpdateResult result;
    
    // 비동기 처리를 위한 CUDA 스트림
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // GPU 메모리 할당 (서브그래프 크기만)
    int *d_row_ptr, *d_col_idx, *d_labels_old, *d_labels_new;
    int *d_local_flags, *d_boundary_indices;
    double *d_penalty;
    
    GPUMemoryManager::safeMalloc((void**)&d_row_ptr, subgraph.row_ptr.size() * sizeof(int));
    GPUMemoryManager::safeMalloc((void**)&d_col_idx, subgraph.col_idx.size() * sizeof(int));
    GPUMemoryManager::safeMalloc((void**)&d_labels_old, subgraph.num_nodes * sizeof(int));
    GPUMemoryManager::safeMalloc((void**)&d_labels_new, subgraph.num_nodes * sizeof(int));
    GPUMemoryManager::safeMalloc((void**)&d_local_flags, subgraph.num_nodes * sizeof(int));
    GPUMemoryManager::safeMalloc((void**)&d_boundary_indices, subgraph.boundary_indices.size() * sizeof(int));
    GPUMemoryManager::safeMalloc((void**)&d_penalty, penalty.size() * sizeof(double));
    
    // 비동기 메모리 전송
    cudaMemcpyAsync(d_row_ptr, subgraph.row_ptr.data(), subgraph.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_col_idx, subgraph.col_idx.data(), subgraph.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_labels_old, subgraph.labels.data(), subgraph.num_nodes * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_labels_new, subgraph.labels.data(), subgraph.num_nodes * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_local_flags, subgraph.local_node_flags.data(), subgraph.num_nodes * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boundary_indices, subgraph.boundary_indices.data(), subgraph.boundary_indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_penalty, penalty.data(), penalty.size() * sizeof(double), cudaMemcpyHostToDevice, stream);
    
    // 커널 실행 설정 (워프 기반 - 각 워프가 하나의 노드 담당)
    const int WARP_SIZE = 32;
    int boundary_count = subgraph.boundary_indices.size();
    int total_warps = boundary_count; // 각 워프가 하나의 바운더리 노드 담당
    int threads_per_block = 256; // 블록당 256개 스레드 (8개 워프)
    int warps_per_block = threads_per_block / WARP_SIZE;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    
    // 공유 메모리 계산: 워프별 파티션 배열 (8워프 x 32파티션)
    size_t shared_mem = 8 * 32 * sizeof(double); // 고정 크기
    
    boundaryLPKernel_unified<<<blocks, threads_per_block, shared_mem, stream>>>(
        d_row_ptr, d_col_idx, d_labels_old, d_labels_new, d_local_flags, d_penalty,
        d_boundary_indices, subgraph.boundary_indices.size(),
        num_partitions, subgraph.num_nodes);
    
    // GPU 커널 실행 완료 대기
    cudaStreamSynchronize(stream);
    
    // 결과 복사
    std::vector<int> updated_labels(subgraph.num_nodes);
    cudaMemcpyAsync(updated_labels.data(), d_labels_new, subgraph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // 변경된 로컬 노드만 추출
    for (int i = 0; i < subgraph.num_nodes; i++) {
        if (subgraph.local_node_flags[i] == 1) { // 로컬 노드만
            if (subgraph.labels[i] != updated_labels[i]) { // 라벨이 변경된 경우
                int orig_node_id = subgraph.node_mapping[i];
                result.updated_nodes.push_back(orig_node_id);
                result.updated_labels.push_back(updated_labels[i]);
                result.change_count++;
            }
        }
    }
    
    // 리소스 정리
    cudaStreamDestroy(stream);
    GPUMemoryManager::safeFree(d_row_ptr);
    GPUMemoryManager::safeFree(d_col_idx);
    GPUMemoryManager::safeFree(d_labels_old);
    GPUMemoryManager::safeFree(d_labels_new);
    GPUMemoryManager::safeFree(d_local_flags);
    GPUMemoryManager::safeFree(d_boundary_indices);
    GPUMemoryManager::safeFree(d_penalty);
    
    return result;
}

/**
 * 스트리밍 방식 GPU 처리 (개선된 인터페이스)
 */
GPULabelUpdateResult runBoundaryLPOnGPU_Streaming(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& boundary_nodes,
    const std::vector<int>& local_labels,
    const std::vector<int>& ghost_labels,
    const std::vector<int>& global_ids,
    const std::vector<double>& penalty,
    int num_local_nodes,
    int num_partitions,
    size_t max_memory_mb)
{
    // 통합 서브그래프 생성
    BoundarySubgraph subgraph = createBoundarySubgraphUnified(
        row_ptr, col_idx, boundary_nodes, local_labels, ghost_labels, global_ids, num_local_nodes);
    
    // 메모리 사용량 계산
    size_t subgraph_memory = (subgraph.row_ptr.size() + subgraph.col_idx.size() + 
                             subgraph.num_nodes * 4) * sizeof(int) + // CSR + labels + flags + boundary_indices 
                            penalty.size() * sizeof(double);
    
    size_t available_memory = max_memory_mb * 1024 * 1024;
    
    if (subgraph_memory <= available_memory) {
        // 전체 서브그래프가 메모리에 들어가는 경우
        return runBoundaryLPOnGPU_SubgraphUnified(subgraph, penalty, num_partitions);
    } else {
        // 청크 단위 처리 필요
        return runBoundaryLPOnGPU_Chunked(subgraph, penalty, num_partitions, available_memory);
    }
}

/**
 * 청크 단위 처리 함수 구현
 * 큰 서브그래프를 메모리에 맞게 청크로 나누어 처리
 */
GPULabelUpdateResult runBoundaryLPOnGPU_Chunked(
    const BoundarySubgraph& subgraph,
    const std::vector<double>& penalty,
    int num_partitions,
    size_t available_memory) {
    
    // 결과 누적용
    GPULabelUpdateResult total_result;
    
    // 각 청크당 최대 노드 수 계산
    size_t per_node_memory = sizeof(int) * 3 + sizeof(int); // CSR + labels + flags
    size_t overhead_memory = penalty.size() * sizeof(double) + 1024 * 1024; // penalty + 1MB 오버헤드
    size_t usable_memory = available_memory - overhead_memory;
    int max_nodes_per_chunk = std::max(1, (int)(usable_memory / per_node_memory));
    
    // 노드 범위별로 청크 처리
    for (int start_node = 0; start_node < subgraph.num_nodes; start_node += max_nodes_per_chunk) {
        int end_node = std::min(start_node + max_nodes_per_chunk, subgraph.num_nodes);
        
        // 청크용 서브그래프 생성
        BoundarySubgraph chunk_subgraph = createChunkSubgraph(subgraph, start_node, end_node);
        
        // 청크 처리
        GPULabelUpdateResult chunk_result = runBoundaryLPOnGPU_SubgraphUnified(
            chunk_subgraph, penalty, num_partitions);
        
        // 결과 병합 (노드 인덱스를 원래 인덱스로 변환)
        for (size_t i = 0; i < chunk_result.updated_nodes.size(); i++) {
            int original_node = chunk_result.updated_nodes[i] + start_node;
            if (original_node < subgraph.num_local_nodes) { // 로컬 노드만
                total_result.updated_nodes.push_back(original_node);
                total_result.updated_labels.push_back(chunk_result.updated_labels[i]);
                total_result.change_count++;
            }
        }
    }
    
    return total_result;
}

/**
 * 청크용 서브그래프 생성
 */
BoundarySubgraph createChunkSubgraph(const BoundarySubgraph& original, int start_node, int end_node) {
    BoundarySubgraph chunk;
    chunk.num_nodes = end_node - start_node;
    chunk.num_local_nodes = 0;
    
    // 로컬 노드 수 계산
    for (int i = start_node; i < end_node; i++) {
        if (original.local_node_flags[i]) {
            chunk.num_local_nodes++;
        }
    }
    
    // 청크의 엣지 수 계산
    chunk.num_edges = 0;
    for (int i = start_node; i < end_node; i++) {
        chunk.num_edges += original.row_ptr[i+1] - original.row_ptr[i];
    }
    
    // 메모리 할당
    chunk.row_ptr.resize(chunk.num_nodes + 1);
    chunk.col_idx.resize(chunk.num_edges);
    chunk.labels.resize(chunk.num_nodes);
    chunk.local_node_flags.resize(chunk.num_nodes);
    
    // 데이터 복사
    chunk.row_ptr[0] = 0;
    int edge_offset = 0;
    
    for (int i = 0; i < chunk.num_nodes; i++) {
        int original_node = start_node + i;
        
        // 라벨과 플래그 복사
        chunk.labels[i] = original.labels[original_node];
        chunk.local_node_flags[i] = original.local_node_flags[original_node];
        
        // 엣지 정보 복사
        int start_edge = original.row_ptr[original_node];
        int end_edge = original.row_ptr[original_node + 1];
        int edge_count = end_edge - start_edge;
        
        for (int j = 0; j < edge_count; j++) {
            chunk.col_idx[edge_offset + j] = 
                original.col_idx[start_edge + j] - start_node; // 상대 인덱스로 변환
        }
        
        edge_offset += edge_count;
        chunk.row_ptr[i + 1] = edge_offset;
    }
    
    return chunk;
}
