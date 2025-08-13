#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>

// 바운더리 서브그래프 구조체
struct BoundarySubgraph {
    std::vector<int> row_ptr;           // CSR row pointers
    std::vector<int> col_idx;           // CSR column indices  
    std::vector<int> node_mapping;      // 서브그래프 인덱스 -> 원본 노드 ID
    std::vector<int> reverse_mapping;   // 원본 노드 ID -> 서브그래프 인덱스 (-1 if not exists)
    std::vector<int> boundary_indices;  // 서브그래프 내 바운더리 노드 인덱스들
    int num_nodes;                      // 서브그래프 노드 수
    int num_edges;                      // 서브그래프 간선 수
    
    BoundarySubgraph() : num_nodes(0), num_edges(0) {}
};

// 스트리밍 처리용 청크 구조체
struct BoundaryChunk {
    BoundarySubgraph subgraph;
    std::vector<int> labels_subset;     // 청크 내 노드들의 라벨
    std::vector<double> penalty_subset; // 청크 내 파티션들의 패널티
    size_t chunk_id;
    size_t total_chunks;
};

// 바운더리 서브그래프 생성 함수
BoundarySubgraph createBoundarySubgraph(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& boundary_nodes,
    const std::vector<int>& labels,
    int labels_count);

// 적응적 바운더리 확장 (이전 바운더리 + 1-hop)
std::vector<int> expandBoundaryNodes(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& prev_boundary_nodes,
    const std::vector<int>& labels,
    int labels_count);

// 스트리밍 방식 바운더리 LP (메모리 효율적)
void runBoundaryLPOnGPU_Streaming(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions,
    size_t max_memory_mb = 512);

// 서브그래프 최적화 처리
void runBoundaryLPOnGPU_SubgraphOptimized(
    const BoundarySubgraph& subgraph,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    int num_partitions);

// 청크 단위 처리
void processSubgraphChunk(
    const BoundarySubgraph& subgraph,
    const std::vector<int>& chunk_boundary_indices,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    int num_partitions,
    cudaStream_t stream,
    size_t chunk_id);

// 고성능 GPU 라벨 전파 함수
void runBoundaryLPOnGPU_Optimized(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<int>& labels_old,
    std::vector<int>& labels_new,
    const std::vector<double>& penalty,
    const std::vector<int>& boundary_nodes,
    int num_partitions);

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