#ifndef DMOLP_MPI_WORKFLOW_H
#define DMOLP_MPI_WORKFLOW_H

#include "types.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <mutex>
#include <memory>

#ifdef USE_CUDA
#include <cuda_runtime.h>
// Forward declaration for GPU memory manager
class GPUMemoryManager;
struct PartitionInfoGPU;
#endif

// MPI 분산 워크플로우 클래스
class MPIDistributedWorkflowV2 {
private:
    int mpi_rank_;
    int mpi_size_;
    int num_partitions_;
    
    // 정점 소유권 정보
    int start_vertex_;  // 이 MPI 프로세스가 소유하는 정점 시작 ID
    int end_vertex_;    // 이 MPI 프로세스가 소유하는 정점 끝 ID (exclusive)
    
    // Phase 1 메트릭 (비교용)
    Phase1Metrics phase1_metrics_;
    
    // 그래프 데이터
    Graph local_graph_;
    std::vector<int> vertex_labels_;  // 각 정점의 라벨 (파티션 ID)
    
    // 7단계 알고리즘 배열들
    std::vector<int> BV_;  // Boundary Vertices (인접리스트 기반)
    std::unordered_map<int, int> NV_;  // Neighbor Vertices (key-value 기반)
    std::vector<PartitionInfo> PI_;  // Partition Information
    PartitionUpdate PU_;  // Partition Update arrays
    
    // 메트릭
    int current_edge_cut_;
    int previous_edge_cut_;
    double edge_rate_;
    int convergence_count_;
    
#ifdef USE_CUDA
    // GPU 메모리 매니저 (Multi-GPU 지원)
    std::vector<std::unique_ptr<GPUMemoryManager>> gpu_managers_;
    int num_gpus_;
    std::vector<cudaDeviceProp> gpu_properties_;
#endif

public:
    MPIDistributedWorkflowV2(int argc, char* argv[], const Graph& phase1_graph, 
                            const std::vector<int>& phase1_labels, const Phase1Metrics& phase1_metrics);
    ~MPIDistributedWorkflowV2();
    
    void run();

private:
    // === 7단계 알고리즘 메소드들 ===
    
    // Step 1: RV, RE 계산
    void calculateRatios();
    
    // Step 2: 전체 imbalance 계산
    void calculateImbalance();
    
    // Step 3: Edge-cut 계산 및 BV/NV 추출
    void calculateEdgeCutAndExtractBoundary();
    void extractBoundaryVerticesWithOpenMP();
    void extractBoundaryVertices();
    
    // Step 4: Dynamic Unweighted LP 수행
    void performDynamicLabelPropagation();
    void updatePartitionInfoOnGPU(); // CPU/CUDA 공통 인터페이스
    int performDynamicLabelPropagationGPU(); // CPU/CUDA 공통 인터페이스
    int performDynamicLabelPropagationCPU();
    
    // Step 5: 진짜 고스트 노드 교환 (선택적 점대점 통신)
    void exchangePartitionUpdates();
    void processGhostRequest(int source_rank, const std::vector<int>& request_buffer);
    int getOwnerRank(int vertex_id) const;
    
    // Step 6: 수렴 확인
    bool checkConvergence();
    
    // Step 7: 다음 반복 준비 + Ghost Node 복제
    void prepareNextIteration();
    void exchangeGhostNodeInformation(std::vector<std::pair<int, int>>& ghost_updates);
    void applyGhostNodeUpdates(const std::vector<std::pair<int, int>>& ghost_updates);
    void updatePartitionBoundaries();
    
    // 최종 결과 출력
    void printFinalResults(long execution_time_ms);
};

#endif // DMOLP_MPI_WORKFLOW_H
