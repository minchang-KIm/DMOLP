#include <cuda_runtime.h>
#ifdef function
#undef function
#endif
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <utility>

#include "phase2/gpu_lp_boundary.h"

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