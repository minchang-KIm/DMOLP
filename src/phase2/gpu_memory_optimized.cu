#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <memory>
#include <unordered_map>

#include "phase2/gpu_lp_boundary.h"

// ==================== 메모리 관리 클래스 ====================
class GPUMemoryManager {
private:
    static std::unordered_map<void*, size_t> allocated_memory;
    static size_t total_allocated;
    static bool leak_detection_enabled;

public:
    static void enableLeakDetection() { leak_detection_enabled = true; }
    static void disableLeakDetection() { leak_detection_enabled = false; }
    
    static cudaError_t safeMalloc(void** ptr, size_t size) {
        cudaError_t err = cudaMalloc(ptr, size);
        if (err == cudaSuccess && leak_detection_enabled) {
            allocated_memory[*ptr] = size;
            total_allocated += size;
            printf("[GPU-Memory] Allocated %zu bytes at %p (total: %zu bytes)\n", 
                   size, *ptr, total_allocated);
        }
        return err;
    }
    
    static cudaError_t safeFree(void* ptr) {
        if (leak_detection_enabled && allocated_memory.find(ptr) != allocated_memory.end()) {
            size_t size = allocated_memory[ptr];
            total_allocated -= size;
            allocated_memory.erase(ptr);
            printf("[GPU-Memory] Freed %zu bytes at %p (total: %zu bytes)\n", 
                   size, ptr, total_allocated);
        }
        return cudaFree(ptr);
    }
    
    static void reportLeaks() {
        if (leak_detection_enabled && !allocated_memory.empty()) {
            printf("[GPU-Memory-LEAK] Found %zu unfreed allocations:\n", allocated_memory.size());
            for (const auto& pair : allocated_memory) {
                printf("  - %p: %zu bytes\n", pair.first, pair.second);
            }
            printf("[GPU-Memory-LEAK] Total leaked: %zu bytes\n", total_allocated);
        } else if (leak_detection_enabled) {
            printf("[GPU-Memory] No memory leaks detected!\n");
        }
    }
    
    static size_t getTotalAllocated() { return total_allocated; }
};

// 정적 멤버 초기화
std::unordered_map<void*, size_t> GPUMemoryManager::allocated_memory;
size_t GPUMemoryManager::total_allocated = 0;
bool GPUMemoryManager::leak_detection_enabled = false;

// ==================== Pinned Memory 최적화 클래스 ====================
class PinnedMemoryPool {
private:
    struct PinnedBuffer {
        void* host_ptr;
        void* device_ptr;
        size_t size;
        bool in_use;
        
        PinnedBuffer(size_t sz) : size(sz), in_use(false) {
            // Pinned Host Memory 할당 (페이지 잠김 메모리)
            cudaMallocHost(&host_ptr, size);
            GPUMemoryManager::safeMalloc(&device_ptr, size);
        }
        
        ~PinnedBuffer() {
            if (host_ptr) cudaFreeHost(host_ptr);
            if (device_ptr) GPUMemoryManager::safeFree(device_ptr);
        }
    };
    
    static std::vector<std::unique_ptr<PinnedBuffer>> buffer_pool;
    static const size_t MAX_POOL_SIZE = 10;  // 최대 10개 버퍼 캐시

public:
    static PinnedBuffer* acquireBuffer(size_t size) {
        // 기존 버퍼에서 적합한 크기 찾기
        for (auto& buffer : buffer_pool) {
            if (!buffer->in_use && buffer->size >= size) {
                buffer->in_use = true;
                printf("[Pinned-Pool] Reusing buffer %p (size: %zu)\n", 
                       buffer->host_ptr, buffer->size);
                return buffer.get();
            }
        }
        
        // 새 버퍼 생성
        if (buffer_pool.size() < MAX_POOL_SIZE) {
            auto new_buffer = std::make_unique<PinnedBuffer>(size);
            printf("[Pinned-Pool] Created new buffer %p (size: %zu)\n", 
                   new_buffer->host_ptr, size);
            new_buffer->in_use = true;
            PinnedBuffer* ptr = new_buffer.get();
            buffer_pool.push_back(std::move(new_buffer));
            return ptr;
        }
        
        // 풀이 가득 찬 경우 임시 버퍼 생성
        printf("[Pinned-Pool] Pool full, creating temporary buffer (size: %zu)\n", size);
        return new PinnedBuffer(size);
    }
    
    static void releaseBuffer(PinnedBuffer* buffer) {
        bool found = false;
        for (auto& pooled_buffer : buffer_pool) {
            if (pooled_buffer.get() == buffer) {
                buffer->in_use = false;
                found = true;
                printf("[Pinned-Pool] Released buffer %p back to pool\n", buffer->host_ptr);
                break;
            }
        }
        
        if (!found) {
            // 임시 버퍼인 경우 삭제
            printf("[Pinned-Pool] Deleting temporary buffer %p\n", buffer->host_ptr);
            delete buffer;
        }
    }
    
    static void clearPool() {
        printf("[Pinned-Pool] Clearing pool (%zu buffers)\n", buffer_pool.size());
        buffer_pool.clear();
    }
    
    static size_t getPoolSize() { return buffer_pool.size(); }
};

// 정적 멤버 초기화
std::vector<std::unique_ptr<PinnedMemoryPool::PinnedBuffer>> PinnedMemoryPool::buffer_pool;

// ==================== CUDA 커널 (메모리 최적화) ====================
__global__ void boundaryLPKernel_memory_optimized(
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

    // 동적 공유 메모리 (최적화된 크기)
    extern __shared__ double scores[];
    
    // 워프 협력적 초기화 (메모리 대역폭 최적화)
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        scores[l] = 0.0;
    }
    __syncthreads();

    // 이웃 노드 처리 (캐시 친화적 접근)
    int start = row_ptr[node];
    int end = row_ptr[node + 1];
    
    // 로컬 레지스터 캐시 사용
    double local_scores[8] = {0.0};  // 최대 8개 파티션 로컬 캐시
    int max_local = min(num_partitions, 8);
    
    for (int e = start; e < end; e++) {
        int neighbor = col_idx[e];
        int neighbor_label = labels_old[neighbor];
        
        if (neighbor_label >= 0 && neighbor_label < num_partitions) {
            if (neighbor_label < max_local) {
                local_scores[neighbor_label] += 1.0;
            } else {
                atomicAdd(&scores[neighbor_label], 1.0);
            }
        }
    }
    
    // 로컬 스코어를 공유 메모리에 합산
    for (int l = 0; l < max_local; l++) {
        if (local_scores[l] > 0.0) {
            atomicAdd(&scores[l], local_scores[l]);
        }
    }
    __syncthreads();
    
    // 패널티 적용 (벡터화)
    for (int l = threadIdx.x; l < num_partitions; l += blockDim.x) {
        if (scores[l] > 0.0) {
            scores[l] = scores[l] * (1.0 + penalty[l]);
        }
    }
    __syncthreads();

    // 최대값 찾기 (워프 최적화)
    int best_label = my_label;
    double best_score = (my_label >= 0 && my_label < num_partitions) ? scores[my_label] : 0.0;
    
    for (int l = 0; l < num_partitions; l++) {
        if (scores[l] > best_score) {
            best_score = scores[l];
            best_label = l;
        }
    }
    
    // 원자적 업데이트
    if (best_label != my_label) {
        atomicExch(&labels_new[node], best_label);
    }
}

// ==================== Pinned Memory 최적화 GPU 함수 ====================
void runBoundaryLPOnGPU_PinnedOptimized(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions)
{
    // 메모리 누수 감지 활성화
    GPUMemoryManager::enableLeakDetection();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // CUDA 스트림 생성 (비동기 최적화)
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 데이터 크기 계산
        size_t row_size = row_ptr.size() * sizeof(int);
        size_t col_size = col_idx.size() * sizeof(int);
        size_t labels_size = labels_old.size() * sizeof(int);
        size_t boundary_size = boundary_nodes.size() * sizeof(int);
        size_t penalty_size = penalty.size() * sizeof(double);
        
        // Pinned Memory 버퍼 획득
        auto row_buffer = PinnedMemoryPool::acquireBuffer(row_size);
        auto col_buffer = PinnedMemoryPool::acquireBuffer(col_size);
        auto labels_old_buffer = PinnedMemoryPool::acquireBuffer(labels_size);
        auto labels_new_buffer = PinnedMemoryPool::acquireBuffer(labels_size);
        auto boundary_buffer = PinnedMemoryPool::acquireBuffer(boundary_size);
        auto penalty_buffer = PinnedMemoryPool::acquireBuffer(penalty_size);
        
        // 호스트 데이터를 Pinned Memory로 복사
        memcpy(row_buffer->host_ptr, row_ptr.data(), row_size);
        memcpy(col_buffer->host_ptr, col_idx.data(), col_size);
        memcpy(labels_old_buffer->host_ptr, labels_old.data(), labels_size);
        memcpy(labels_new_buffer->host_ptr, labels_new.data(), labels_size);
        memcpy(boundary_buffer->host_ptr, boundary_nodes.data(), boundary_size);
        memcpy(penalty_buffer->host_ptr, penalty.data(), penalty_size);
        
        // 비동기 메모리 전송 (Pinned Memory -> GPU)
        cudaMemcpyAsync(row_buffer->device_ptr, row_buffer->host_ptr, row_size, 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(col_buffer->device_ptr, col_buffer->host_ptr, col_size, 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(labels_old_buffer->device_ptr, labels_old_buffer->host_ptr, labels_size, 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(labels_new_buffer->device_ptr, labels_new_buffer->host_ptr, labels_size, 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(boundary_buffer->device_ptr, boundary_buffer->host_ptr, boundary_size, 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(penalty_buffer->device_ptr, penalty_buffer->host_ptr, penalty_size, 
                       cudaMemcpyHostToDevice, stream);
        
        // 커널 실행 설정 (메모리 최적화)
        int blockSize = 256;
        int gridSize = (boundary_nodes.size() + blockSize - 1) / blockSize;
        size_t shared_mem = (num_partitions + 32) * sizeof(double);  // 패딩 추가
        
        // 최적화된 커널 실행
        boundaryLPKernel_memory_optimized<<<gridSize, blockSize, shared_mem, stream>>>(
            (int*)row_buffer->device_ptr, (int*)col_buffer->device_ptr,
            (int*)labels_old_buffer->device_ptr, (int*)labels_new_buffer->device_ptr,
            (double*)penalty_buffer->device_ptr,
            (int*)boundary_buffer->device_ptr, boundary_nodes.size(),
            num_partitions);
        
        // 결과를 Pinned Memory로 비동기 복사
        cudaMemcpyAsync(labels_new_buffer->host_ptr, labels_new_buffer->device_ptr, labels_size, 
                       cudaMemcpyDeviceToHost, stream);
        
        // 스트림 동기화
        cudaStreamSynchronize(stream);
        
        // 결과를 최종 벡터로 복사
        memcpy(labels_new.data(), labels_new_buffer->host_ptr, labels_size);
        
        // CUDA 오류 검사
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("[GPU-Pinned-Error] CUDA error: %s\n", cudaGetErrorString(error));
        }
        
        // 리소스 정리
        cudaStreamDestroy(stream);
        
        // Pinned 버퍼 해제
        PinnedMemoryPool::releaseBuffer(row_buffer);
        PinnedMemoryPool::releaseBuffer(col_buffer);
        PinnedMemoryPool::releaseBuffer(labels_old_buffer);
        PinnedMemoryPool::releaseBuffer(labels_new_buffer);
        PinnedMemoryPool::releaseBuffer(boundary_buffer);
        PinnedMemoryPool::releaseBuffer(penalty_buffer);
        
    } catch (const std::exception& e) {
        printf("[GPU-Pinned-Exception] %s\n", e.what());
    }
    
    // 성능 측정
    auto end_time = std::chrono::high_resolution_clock::now();
    auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    printf("[GPU-Pinned-Optimized] Execution time: %ld μs (boundary nodes: %zu)\n", 
           exec_time, boundary_nodes.size());
    printf("[GPU-Pinned-Pool] Pool size: %zu buffers\n", PinnedMemoryPool::getPoolSize());
    
    // 메모리 누수 리포트
    GPUMemoryManager::reportLeaks();
}

// ==================== 정리 함수 ====================
void cleanupGPUResources() {
    printf("[GPU-Cleanup] Cleaning up all GPU resources...\n");
    PinnedMemoryPool::clearPool();
    GPUMemoryManager::reportLeaks();
    printf("[GPU-Cleanup] Cleanup complete.\n");
}
