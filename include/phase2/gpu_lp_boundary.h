#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>

// 바운더리 서브그래프 구조체 (GPU 전용)
struct BoundarySubgraph {
    std::vector<int> row_ptr;           // CSR row pointers
    std::vector<int> col_idx;           // CSR column indices  
    std::vector<int> node_mapping;      // 서브그래프 인덱스 -> 원본 노드 ID
    std::vector<int> reverse_mapping;   // 원본 노드 ID -> 서브그래프 인덱스 (-1 if not exists)
    std::vector<int> boundary_indices;  // 서브그래프 내 바운더리 노드 인덱스들
    std::vector<int> local_node_flags;  // 서브그래프 내 로컬 노드 여부 (1=로컬, 0=고스트)
    std::vector<int> labels;            // 서브그래프 내 모든 노드의 라벨 (로컬+고스트 통합)
    int num_nodes;                      // 서브그래프 노드 수
    int num_edges;                      // 서브그래프 간선 수
    int num_local_nodes;                // 로컬 노드 수 (원본 그래프에서)
    
    BoundarySubgraph() : num_nodes(0), num_edges(0), num_local_nodes(0) {}
};

// GPU 전용 라벨 업데이트 결과
struct GPULabelUpdateResult {
    std::vector<int> updated_nodes;     // 업데이트된 로컬 노드들 (원본 ID)
    std::vector<int> updated_labels;    // 새로운 라벨들
    int change_count;                   // 변경된 노드 수
    
    GPULabelUpdateResult() : change_count(0) {}
};

// 바운더리 서브그래프 생성 함수 (로컬+고스트 통합)
BoundarySubgraph createBoundarySubgraphUnified(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& boundary_nodes,
    const std::vector<int>& local_labels,
    const std::vector<int>& ghost_labels,
    const std::vector<int>& global_ids,
    int num_local_nodes);

// 적응적 바운더리 확장 (이전 바운더리 + 1-hop)
std::vector<int> expandBoundaryNodes(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& prev_boundary_nodes,
    const std::vector<int>& labels,
    const std::vector<double>& penalty,
    const std::vector<double>& RE,
    int labels_count,
    int iter);

// 효율적인 GPU 라벨 전파 (서브그래프만 전달, 로컬만 업데이트)
GPULabelUpdateResult runBoundaryLPOnGPU_SubgraphUnified(
    const BoundarySubgraph& subgraph,
    const std::vector<double>& penalty,
    int num_partitions);

// 청크 단위 처리 (큰 서브그래프용)
GPULabelUpdateResult runBoundaryLPOnGPU_Chunked(
    const BoundarySubgraph& subgraph,
    const std::vector<double>& penalty,
    int num_partitions,
    size_t available_memory);

// 청크 관련 헬퍼 함수들
BoundarySubgraph createChunkSubgraph(const BoundarySubgraph& original, int start_node, int end_node);
void freeChunkSubgraph(BoundarySubgraph& chunk);

// 스트리밍 방식 바운더리 LP (메모리 효율적)
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
    size_t max_memory_mb = 512);

// Thrust 버전은 제거되었습니다 (단일 고성능/핀드/세이프 경로만 유지)

// 안전한 GPU 라벨 전파 함수 (fallback)
void runBoundaryLPOnGPU_Safe(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions);

// Pinned Memory 최적화 GPU 라벨 전파 함수 (메모리 누수 방지 + 성능 최적화)
void runBoundaryLPOnGPU_PinnedOptimized(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions);

// GPU 리소스 정리 함수
void cleanupGPUResources();