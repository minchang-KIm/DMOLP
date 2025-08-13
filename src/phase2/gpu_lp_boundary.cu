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

#include "phase2/gpu_lp_boundary.h"

// ==================== 바운더리 서브그래프 생성 (로컬+고스트 통합) ====================

/**
 * 바운더리 노드 + 1-hop 이웃으로 구성된 통합 서브그래프 생성
 * - 로컬 노드와 고스트 노드의 라벨을 통합하여 GPU에 전달
 * - 로컬 노드만 업데이트 대상으로 표시
 * - GPU 메모리 지역성 최적화
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
    
    // 1단계: 서브그래프에 포함될 모든 노드 수집 (바운더리 + 1-hop 이웃)
    std::unordered_set<int> subgraph_nodes_set;
    
    // 바운더리 노드들 추가
    for (int boundary_node : boundary_nodes) {
        if (boundary_node >= 0 && boundary_node < (int)(local_labels.size() + ghost_labels.size())) {
            subgraph_nodes_set.insert(boundary_node);
        }
    }
    
    // 각 바운더리 노드의 1-hop 이웃 추가
    for (int boundary_node : boundary_nodes) {
        if (boundary_node >= 0 && boundary_node < (int)row_ptr.size() - 1) {
            for (int edge_idx = row_ptr[boundary_node]; edge_idx < row_ptr[boundary_node + 1]; edge_idx++) {
                int neighbor = col_idx[edge_idx];
                if (neighbor >= 0 && neighbor < (int)(local_labels.size() + ghost_labels.size())) {
                    subgraph_nodes_set.insert(neighbor);
                }
            }
        }
    }
    
    // 2단계: 노드 매핑 구성
    std::vector<int> subgraph_nodes(subgraph_nodes_set.begin(), subgraph_nodes_set.end());
    std::sort(subgraph_nodes.begin(), subgraph_nodes.end());
    
    subgraph.num_nodes = subgraph_nodes.size();
    subgraph.node_mapping = subgraph_nodes;
    subgraph.reverse_mapping.resize(local_labels.size() + ghost_labels.size(), -1);
    
    // 통합 라벨 배열 및 로컬 노드 플래그 구성
    subgraph.labels.resize(subgraph.num_nodes);
    subgraph.local_node_flags.resize(subgraph.num_nodes);
    
    for (int i = 0; i < subgraph.num_nodes; i++) {
        int orig_node = subgraph_nodes[i];
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
    
    // 3단계: 서브그래프 CSR 구성
    subgraph.row_ptr.resize(subgraph.num_nodes + 1, 0);
    std::vector<std::vector<int>> adj_list(subgraph.num_nodes);
    
    // 각 서브그래프 노드에 대해 이웃 수집
    for (int i = 0; i < subgraph.num_nodes; i++) {
        int orig_node = subgraph_nodes[i];
        if (orig_node < (int)row_ptr.size() - 1) {
            for (int edge_idx = row_ptr[orig_node]; edge_idx < row_ptr[orig_node + 1]; edge_idx++) {
                int neighbor = col_idx[edge_idx];
                if (neighbor >= 0 && neighbor < (int)(local_labels.size() + ghost_labels.size())) {
                    int neighbor_subgraph_idx = subgraph.reverse_mapping[neighbor];
                    if (neighbor_subgraph_idx != -1) {
                        adj_list[i].push_back(neighbor_subgraph_idx);
                    }
                }
            }
        }
    }
    
    // CSR 형태로 변환
    int edge_count = 0;
    for (int i = 0; i < subgraph.num_nodes; i++) {
        subgraph.row_ptr[i] = edge_count;
        edge_count += adj_list[i].size();
    }
    subgraph.row_ptr[subgraph.num_nodes] = edge_count;
    subgraph.num_edges = edge_count;
    
    subgraph.col_idx.resize(edge_count);
    int idx = 0;
    for (int i = 0; i < subgraph.num_nodes; i++) {
        for (int neighbor : adj_list[i]) {
            subgraph.col_idx[idx++] = neighbor;
        }
    }
    
    // 4단계: 서브그래프 내 실제 바운더리 노드 인덱스 찾기 (로컬 노드만)
    std::unordered_set<int> boundary_set(boundary_nodes.begin(), boundary_nodes.end());
    for (int i = 0; i < subgraph.num_nodes; i++) {
        int orig_node = subgraph_nodes[i];
        if (boundary_set.count(orig_node) && subgraph.local_node_flags[i] == 1) {
            subgraph.boundary_indices.push_back(i);
        }
    }
    
    int local_count = 0;
    for (int flag : subgraph.local_node_flags) {
        if (flag == 1) local_count++;
    }
    
    printf("[Subgraph-Unified] Created: %d nodes (%d local, %d ghost), %d edges, %zu boundary nodes\n", 
           subgraph.num_nodes, local_count, subgraph.num_nodes - local_count, 
           subgraph.num_edges, subgraph.boundary_indices.size());
    
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
    
    printf("[Boundary-Expansion] %zu -> %zu nodes\n", 
           prev_boundary_nodes.size(), new_boundary_nodes.size());
    
    return new_boundary_nodes;
}

// ==================== 메모리 관리 및 핀드 메모리 풀 ====================
class GPUMemoryManager {
private:
    static std::vector<std::pair<void*, size_t>> allocations;
    static size_t total_allocated;
    static bool leak_detection_enabled;

public:
    static void enableLeakDetection() { leak_detection_enabled = true; }
    static void disableLeakDetection() { leak_detection_enabled = false; }
    
    static cudaError_t safeMalloc(void** ptr, size_t size) {
        cudaError_t err = cudaMalloc(ptr, size);
        if (err == cudaSuccess && leak_detection_enabled) {
            allocations.emplace_back(*ptr, size);
            total_allocated += size;
            printf("[GPU-Memory] Allocated %zu bytes at %p (total: %zu bytes)\n", 
                   size, *ptr, total_allocated);
        }
        return err;
    }
    
    static cudaError_t safeFree(void* ptr) {
        if (leak_detection_enabled) {
            for (auto it = allocations.begin(); it != allocations.end(); ++it) {
                if (it->first == ptr) {
                    size_t size = it->second;
                    total_allocated -= size;
                    allocations.erase(it);
                    printf("[GPU-Memory] Freed %zu bytes at %p (total: %zu bytes)\n", 
                           size, ptr, total_allocated);
                    break;
                }
            }
        }
        return cudaFree(ptr);
    }
    
    static void reportLeaks() {
        if (leak_detection_enabled && !allocations.empty()) {
            printf("[GPU-Memory-LEAK] Found %zu unfreed allocations:\n", allocations.size());
            for (const auto& p : allocations) {
                printf("  - %p: %zu bytes\n", p.first, p.second);
            }
            printf("[GPU-Memory-LEAK] Total leaked: %zu bytes\n", total_allocated);
        } else if (leak_detection_enabled) {
            printf("[GPU-Memory] No memory leaks detected!\n");
        }
    }
    
    static size_t getTotalAllocated() { return total_allocated; }
};

std::vector<std::pair<void*, size_t>> GPUMemoryManager::allocations;
size_t GPUMemoryManager::total_allocated = 0;
bool GPUMemoryManager::leak_detection_enabled = false;

class PinnedMemoryPool {
private:
    struct PinnedBuffer {
        void* host_ptr;
        void* device_ptr;
        size_t size;
        bool in_use;
        
        PinnedBuffer(size_t sz) : host_ptr(nullptr), device_ptr(nullptr), size(sz), in_use(false) {
            cudaMallocHost(&host_ptr, size);
            GPUMemoryManager::safeMalloc(&device_ptr, size);
        }
        
        ~PinnedBuffer() {
            if (host_ptr) cudaFreeHost(host_ptr);
            if (device_ptr) GPUMemoryManager::safeFree(device_ptr);
        }
    };
    
    static std::vector<PinnedBuffer*> buffer_pool;
    static const size_t MAX_POOL_SIZE = 10;

public:
    static PinnedBuffer* acquireBuffer(size_t size) {
        for (auto* buffer : buffer_pool) {
            if (!buffer->in_use && buffer->size >= size) {
                buffer->in_use = true;
                printf("[Pinned-Pool] Reusing buffer %p (size: %zu)\n", 
                       buffer->host_ptr, buffer->size);
                return buffer;
            }
        }
        if (buffer_pool.size() < MAX_POOL_SIZE) {
            PinnedBuffer* new_buffer = new PinnedBuffer(size);
            printf("[Pinned-Pool] Created new buffer %p (size: %zu)\n", 
                   new_buffer->host_ptr, size);
            new_buffer->in_use = true;
            buffer_pool.push_back(new_buffer);
            return new_buffer;
        }
        printf("[Pinned-Pool] Pool full, creating temporary buffer (size: %zu)\n", size);
        return new PinnedBuffer(size);
    }
    
    static void releaseBuffer(PinnedBuffer* buffer) {
        bool found = false;
        for (auto* pooled_buffer : buffer_pool) {
            if (pooled_buffer == buffer) {
                pooled_buffer->in_use = false;
                found = true;
                printf("[Pinned-Pool] Released buffer %p back to pool\n", buffer->host_ptr);
                break;
            }
        }
        if (!found) {
            printf("[Pinned-Pool] Deleting temporary buffer %p\n", buffer->host_ptr);
            delete buffer;
        }
    }
    
    static void clearPool() {
        printf("[Pinned-Pool] Clearing pool (%zu buffers)\n", buffer_pool.size());
        for (auto* buf : buffer_pool) {
            delete buf;
        }
        buffer_pool.clear();
    }
    
    static size_t getPoolSize() { return buffer_pool.size(); }
};

std::vector<PinnedMemoryPool::PinnedBuffer*> PinnedMemoryPool::buffer_pool;

// ==================== 서브그래프 전용 커널 (로컬 노드만 업데이트) ====================

/**
 * 통합 서브그래프 전용 최적화 커널
 * - 로컬+고스트 라벨을 모두 참조하지만 로컬 노드만 업데이트
 * - 로컬 노드 플래그로 업데이트 대상 구분
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundary_count) return;

    int subgraph_node_idx = boundary_indices[idx];
    if (subgraph_node_idx < 0 || subgraph_node_idx >= subgraph_size) return;
    
    // 로컬 노드가 아니면 스킵 (고스트 노드는 업데이트하지 않음)
    if (local_node_flags[subgraph_node_idx] != 1) return;
    
    int my_label = labels_old[subgraph_node_idx];

    extern __shared__ double scores[];
    
    // 공유 메모리 초기화 (워프 협력)
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        scores[l] = 0.0;
    }
    __syncthreads();

    // 로컬 카운터 (레지스터 사용)
    double local_counts[32] = {0.0};
    int max_partitions = (num_partitions < 32 ? num_partitions : 32);
    
    // 이웃 노드 순회 (서브그래프 내에서 로컬+고스트 모두 참조)
    int start = row_ptr[subgraph_node_idx];
    int end = row_ptr[subgraph_node_idx + 1];
    
    for (int e = start; e < end; e++) {
        int neighbor_idx = col_idx[e];
        if (neighbor_idx >= 0 && neighbor_idx < subgraph_size) {
            int neighbor_label = labels_old[neighbor_idx];
            if (neighbor_label >= 0 && neighbor_label < max_partitions) {
                local_counts[neighbor_label] += 1.0;
            }
        }
    }
    
    // 공유 메모리에 합산
    for (int l = 0; l < max_partitions; l++) {
        if (local_counts[l] > 0.0) {
            atomicAdd(&scores[l], local_counts[l]);
        }
    }
    __syncthreads();
    
    // 패널티 적용
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        if (scores[l] > 0.0) {
            scores[l] = scores[l] * (1.0 + penalty[l]);
        }
    }
    __syncthreads();

    // 최적 라벨 찾기
    int best_label = my_label;
    double best_score = (my_label >= 0 && my_label < num_partitions) ? scores[my_label] : 0.0;
    
    for (int l = 0; l < num_partitions; l++) {
        if (scores[l] > best_score) {
            best_score = scores[l];
            best_label = l;
        }
    }
    
    // 결과 저장 (로컬 노드만)
    labels_new[subgraph_node_idx] = best_label;
}

// ==================== 효율적인 GPU 처리 함수 ====================

/**
 * 효율적인 GPU 라벨 전파 (통합 서브그래프, 로컬만 업데이트)
 * - 전체 그래프 대신 서브그래프만 GPU에 전달
 * - 로컬+고스트 라벨 통합으로 GPU 메모리 지역성 최적화
 * - 로컬 노드만 업데이트하여 정확성 보장
 */
GPULabelUpdateResult runBoundaryLPOnGPU_SubgraphUnified(
    const BoundarySubgraph& subgraph,
    const std::vector<double>& penalty,
    int num_partitions)
{
    GPULabelUpdateResult result;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
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
    
    // 커널 실행 설정 (GPU 하드웨어 적응형)
    int threads = 256;
    int boundary_count = subgraph.boundary_indices.size();
    
    // GPU 멀티프로세서 정보 가져오기
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // 최적 블록 개수 계산 (SM 개수와 바운더리 노드 개수 고려)
    int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / threads;
    int max_blocks = prop.multiProcessorCount * max_blocks_per_sm;
    int min_blocks_needed = (boundary_count + threads - 1) / threads;
    int blocks = std::min(max_blocks, min_blocks_needed);
    
    printf("[GPU] SM개수: %d, 최대블록: %d, 필요블록: %d, 사용블록: %d (바운더리: %d)\n",
           prop.multiProcessorCount, max_blocks, min_blocks_needed, blocks, boundary_count);
    
    size_t shared_mem = num_partitions * sizeof(double);
    
    boundaryLPKernel_unified<<<blocks, threads, shared_mem, stream>>>(
        d_row_ptr, d_col_idx, d_labels_old, d_labels_new, d_local_flags, d_penalty,
        d_boundary_indices, subgraph.boundary_indices.size(),
        num_partitions, subgraph.num_nodes);
    
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
    
    // 정리
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("[GPU-Unified] Execution time: %.2f ms (%d nodes, %d boundary, %d changes)\n", 
           ms, subgraph.num_nodes, (int)subgraph.boundary_indices.size(), result.change_count);
    
    cudaStreamDestroy(stream);
    GPUMemoryManager::safeFree(d_row_ptr);
    GPUMemoryManager::safeFree(d_col_idx);
    GPUMemoryManager::safeFree(d_labels_old);
    GPUMemoryManager::safeFree(d_labels_new);
    GPUMemoryManager::safeFree(d_local_flags);
    GPUMemoryManager::safeFree(d_boundary_indices);
    GPUMemoryManager::safeFree(d_penalty);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
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
    printf("[GPU-Streaming] Starting with %zu boundary nodes, memory limit: %zu MB\n", 
           boundary_nodes.size(), max_memory_mb);
    
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
        printf("[GPU-Streaming] Processing entire subgraph (%zu bytes)\n", subgraph_memory);
        return runBoundaryLPOnGPU_SubgraphUnified(subgraph, penalty, num_partitions);
    } else {
        // 청크 단위 처리 필요
        printf("[GPU-Streaming] Subgraph too large (%zu bytes), using chunking\n", subgraph_memory);
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
    size_t per_node_memory = sizeof(int) * 3 + sizeof(bool); // CSR + labels + flags
    size_t overhead_memory = penalty.size() * sizeof(double) + 1024 * 1024; // penalty + 1MB 오버헤드
    size_t usable_memory = available_memory - overhead_memory;
    int max_nodes_per_chunk = std::max(1, (int)(usable_memory / per_node_memory));
    
    printf("[GPU-Chunked] Processing %d nodes in chunks of max %d nodes\n", 
           subgraph.num_nodes, max_nodes_per_chunk);
    
    // 노드 범위별로 청크 처리
    for (int start_node = 0; start_node < subgraph.num_nodes; start_node += max_nodes_per_chunk) {
        int end_node = std::min(start_node + max_nodes_per_chunk, subgraph.num_nodes);
        int chunk_size = end_node - start_node;
        
        printf("[GPU-Chunked] Processing chunk [%d-%d] (%d nodes)\n", 
               start_node, end_node-1, chunk_size);
        
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
            }
        }
        
        // 청크 메모리 해제
        freeChunkSubgraph(chunk_subgraph);
    }
    
    printf("[GPU-Chunked] Total updated nodes: %zu\n", total_result.updated_nodes.size());
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

/**
 * 청크 서브그래프 메모리 해제
 */
void freeChunkSubgraph(BoundarySubgraph& chunk) {
    // std::vector는 자동으로 해제되므로 특별한 작업 불필요
    chunk.row_ptr.clear();
    chunk.col_idx.clear();
    chunk.labels.clear();
    chunk.local_node_flags.clear();
}

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

// ==================== CUDA 커널 (고성능 워프 최적화 - 기본) ====================
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
    int num_partitions,
    int labels_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundary_count) return;

    int node = boundary_nodes[idx];
    if (node < 0 || node >= labels_count) return;
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
    int max_partitions = (num_partitions < 32 ? num_partitions : 32);
    
    // 이웃 노드 순회 및 카운팅
    int start = row_ptr[node];
    int end = row_ptr[node + 1];
    
    for (int e = start; e < end; e++) {
        int neighbor = col_idx[e];
        if (neighbor >= 0 && neighbor < labels_count) {
            int neighbor_label = labels_old[neighbor];
            if (neighbor_label >= 0 && neighbor_label < max_partitions) {
                local_counts[neighbor_label] += 1.0;
            }
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

// ==================== CUDA 커널 (안전 버전) ====================
__global__ void boundaryLPKernel_safe(
    const int* __restrict__ row_ptr, 
    const int* __restrict__ col_idx,
    const int* __restrict__ labels_old, 
    int* __restrict__ labels_new,
    const double* __restrict__ penalty,
    const int* __restrict__ boundary_nodes, 
    int boundary_count,
    int num_partitions,
    int labels_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundary_count) return;

    int node = boundary_nodes[idx];
    if (node < 0 || node >= labels_count) return;
    int my_label = labels_old[node];

    extern __shared__ double scores[];
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        scores[l] = 0.0;
    }
    __syncthreads();

    int start = row_ptr[node];
    int end = row_ptr[node + 1];
    for (int e = start; e < end; e++) {
        int neighbor = col_idx[e];
        if (neighbor >= 0 && neighbor < labels_count) {
            int neighbor_label = labels_old[neighbor];
            if (neighbor_label >= 0 && neighbor_label < num_partitions) {
                atomicAdd(&scores[neighbor_label], 1.0);
            }
        }
    }
    __syncthreads();

    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        if (scores[l] > 0.0) {
            scores[l] = scores[l] * (1.0 + penalty[l]);
        }
    }
    __syncthreads();

    int best_label = my_label;
    double best_score = (my_label >= 0 && my_label < num_partitions) ? scores[my_label] : 0.0;
    for (int l = 0; l < num_partitions; l++) {
        if (scores[l] > best_score) {
            best_score = scores[l];
            best_label = l;
        }
    }
    if (best_label != my_label) {
        labels_new[node] = best_label;
    }
}

// ==================== CUDA 커널 (워프당 1-노드, 핀드 최적화 전용) ====================
__global__ void boundaryLPKernel_memory_optimized_warp(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const int* __restrict__ labels_old,
    int* __restrict__ labels_new,
    const double* __restrict__ penalty,
    const int* __restrict__ boundary_nodes,
    int boundary_count,
    int num_partitions,
    int labels_count)
{
    const int lane = threadIdx.x & 31;
    const int warp_id_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int node_idx = blockIdx.x * warps_per_block + warp_id_in_block;
    if (node_idx >= boundary_count) return;

    const int node = boundary_nodes[node_idx];
    if (node < 0 || node >= labels_count) return;
    const int my_label = labels_old[node];

    extern __shared__ double shm[];
    double* scores = shm + warp_id_in_block * num_partitions;

    for (int l = lane; l < num_partitions; l += 32) {
        scores[l] = 0.0;
    }
    __syncwarp();

    const int start = row_ptr[node];
    const int end   = row_ptr[node + 1];
    for (int e = start + lane; e < end; e += 32) {
        const int nei = col_idx[e];
        if (nei >= 0 && nei < labels_count) {
            const int lbl = labels_old[nei];
            if (lbl >= 0 && lbl < num_partitions) {
                atomicAdd(&scores[lbl], 1.0);
            }
        }
    }
    __syncwarp();

    for (int l = lane; l < num_partitions; l += 32) {
        if (scores[l] > 0.0) {
            scores[l] = scores[l] * (1.0 + penalty[l]);
        }
    }
    __syncwarp();

    if (lane == 0) {
        int best_label = my_label;
        double best_score = (my_label >= 0 && my_label < num_partitions) ? scores[my_label] : 0.0;
        for (int l = 0; l < num_partitions; l++) {
            if (scores[l] > best_score) {
                best_score = scores[l];
                best_label = l;
            }
        }
        if (best_label != my_label) {
            labels_new[node] = best_label;
        }
    }
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
    // 성능 측정 (CUDA 이벤트)
    cudaEvent_t ev_start, ev_stop; float ms=0.0f; cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);
    
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
        num_partitions,
        static_cast<int>(labels_old.size()));

    // 결과 복사
    cudaMemcpyAsync(labels_new.data(), d_labels_new, labels_size, 
                    cudaMemcpyDeviceToHost, stream);
    
    // 동기화 및 시간 측정 종료
    cudaStreamSynchronize(stream);
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop); cudaEventElapsedTime(&ms, ev_start, ev_stop);
    
    printf("[GPU-Optimized] Execution time: %ld μs (boundary nodes: %zu)\n", 
           (long)(ms*1000.0f), boundary_nodes.size());
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);

    // 정리
    cudaStreamDestroy(stream);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_labels_old);
    cudaFree(d_labels_new);
    cudaFree(d_boundary);
    cudaFree(d_penalty);
}

// 안전한 GPU 라벨 전파 (fallback)
void runBoundaryLPOnGPU_Safe(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions)
{
    cudaEvent_t ev_start, ev_stop; float ms=0.0f; cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);
    cudaStream_t stream; cudaStreamCreate(&stream);

    int *d_row_ptr, *d_col_idx, *d_labels_old, *d_labels_new, *d_boundary; double* d_penalty;
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

    cudaMemcpyAsync(d_row_ptr, row_ptr.data(), row_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_col_idx, col_idx.data(), col_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_labels_old, labels_old.data(), labels_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_labels_new, labels_new.data(), labels_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boundary, boundary_nodes.data(), boundary_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_penalty, penalty.data(), penalty_size, cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (boundary_nodes.size() + threads - 1) / threads;
    size_t shared_mem = (num_partitions + 8) * sizeof(double);

    boundaryLPKernel_safe<<<blocks, threads, shared_mem, stream>>>(
        d_row_ptr, d_col_idx, d_labels_old, d_labels_new, d_penalty, d_boundary,
        boundary_nodes.size(), num_partitions,
        static_cast<int>(labels_old.size()));

    cudaMemcpyAsync(labels_new.data(), d_labels_new, labels_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop); cudaEventElapsedTime(&ms, ev_start, ev_stop);
    printf("[GPU-Safe] Execution time: %ld μs (boundary nodes: %zu)\n", (long)(ms*1000.0f), boundary_nodes.size());
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);

    cudaStreamDestroy(stream);
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_labels_old);
    cudaFree(d_labels_new); cudaFree(d_boundary); cudaFree(d_penalty);
}

// 핀드 메모리 최적화 경계 LP (워프당 1-노드)
void runBoundaryLPOnGPU_PinnedOptimized(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions)
{
    GPUMemoryManager::enableLeakDetection();
    cudaEvent_t ev_start, ev_stop; float ms=0.0f; cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);
    
    try {
        cudaStream_t stream; cudaStreamCreate(&stream);
        
        size_t row_size = row_ptr.size() * sizeof(int);
        size_t col_size = col_idx.size() * sizeof(int);
        size_t labels_size = labels_old.size() * sizeof(int);
        size_t boundary_size = boundary_nodes.size() * sizeof(int);
        size_t penalty_size = penalty.size() * sizeof(double);
        
        auto row_buffer = PinnedMemoryPool::acquireBuffer(row_size);
        auto col_buffer = PinnedMemoryPool::acquireBuffer(col_size);
        auto labels_old_buffer = PinnedMemoryPool::acquireBuffer(labels_size);
        auto labels_new_buffer = PinnedMemoryPool::acquireBuffer(labels_size);
        auto boundary_buffer = PinnedMemoryPool::acquireBuffer(boundary_size);
        auto penalty_buffer = PinnedMemoryPool::acquireBuffer(penalty_size);
        
        memcpy(row_buffer->host_ptr, row_ptr.data(), row_size);
        memcpy(col_buffer->host_ptr, col_idx.data(), col_size);
        memcpy(labels_old_buffer->host_ptr, labels_old.data(), labels_size);
        memcpy(labels_new_buffer->host_ptr, labels_new.data(), labels_size);
        memcpy(boundary_buffer->host_ptr, boundary_nodes.data(), boundary_size);
        memcpy(penalty_buffer->host_ptr, penalty.data(), penalty_size);
        
        cudaMemcpyAsync(row_buffer->device_ptr, row_buffer->host_ptr, row_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(col_buffer->device_ptr, col_buffer->host_ptr, col_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(labels_old_buffer->device_ptr, labels_old_buffer->host_ptr, labels_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(labels_new_buffer->device_ptr, labels_new_buffer->host_ptr, labels_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(boundary_buffer->device_ptr, boundary_buffer->host_ptr, boundary_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(penalty_buffer->device_ptr, penalty_buffer->host_ptr, penalty_size, cudaMemcpyHostToDevice, stream);
        
        int warpsPerBlock = 8;
        int blockSize = warpsPerBlock * 32;
        int gridSize = (static_cast<int>(boundary_nodes.size()) + warpsPerBlock - 1) / warpsPerBlock;
        size_t shared_mem = static_cast<size_t>(num_partitions) * warpsPerBlock * sizeof(double);

        boundaryLPKernel_memory_optimized_warp<<<gridSize, blockSize, shared_mem, stream>>>(
            (int*)row_buffer->device_ptr, (int*)col_buffer->device_ptr,
            (int*)labels_old_buffer->device_ptr, (int*)labels_new_buffer->device_ptr,
            (double*)penalty_buffer->device_ptr,
            (int*)boundary_buffer->device_ptr, boundary_nodes.size(),
            num_partitions,
            static_cast<int>(labels_old.size()));
        
        cudaMemcpyAsync(labels_new_buffer->host_ptr, labels_new_buffer->device_ptr, labels_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
        memcpy(labels_new.data(), labels_new_buffer->host_ptr, labels_size);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("[GPU-Pinned-Error] CUDA error: %s\n", cudaGetErrorString(error));
        }
        
        cudaStreamDestroy(stream);
        PinnedMemoryPool::releaseBuffer(row_buffer);
        PinnedMemoryPool::releaseBuffer(col_buffer);
        PinnedMemoryPool::releaseBuffer(labels_old_buffer);
        PinnedMemoryPool::releaseBuffer(labels_new_buffer);
        PinnedMemoryPool::releaseBuffer(boundary_buffer);
        PinnedMemoryPool::releaseBuffer(penalty_buffer);
        
    } catch (const std::exception& e) {
        printf("[GPU-Pinned-Exception] %s\n", e.what());
    }
    
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop); cudaEventElapsedTime(&ms, ev_start, ev_stop);
    printf("[GPU-Pinned-Optimized] Execution time: %ld μs (boundary nodes: %zu)\n", (long)(ms*1000.0f), boundary_nodes.size());
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    printf("[GPU-Pinned-Pool] Pool size: %zu buffers\n", PinnedMemoryPool::getPoolSize());
    
}

// 리소스 정리
void cleanupGPUResources() {
    printf("[GPU-Cleanup] Cleaning up all GPU resources...\n");
    PinnedMemoryPool::clearPool();
    GPUMemoryManager::reportLeaks();
    printf("[GPU-Cleanup] Cleanup complete.\n");
}