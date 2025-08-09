#ifndef DMOLP_GRAPH_TYPES_H
#define DMOLP_GRAPH_TYPES_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>

struct BFSResult {
    std::vector<boost::dynamic_bitset<>> levels;
    boost::dynamic_bitset<> all_visited;

    BFSResult(size_t num_nodes = 0) : all_visited(num_nodes) {}

    void clear() {
        levels.clear();
        all_visited.clear();
    }

    void ensure_level(int level, size_t num_nodes) {
        if (level >= static_cast<int>(levels.size())) levels.resize(level + 1, boost::dynamic_bitset<>(num_nodes));
    }
};

struct Delta {
    int gid;        // global id
    int new_label;  // new label
};

struct NodeInfo {
    int vertex;
    std::vector<int> neighbors;
};

struct FrontierNode {
    int vertex;
    int partition_id;
    double ratio;
    int partition_degree;
    int total_degree;

    FrontierNode() : vertex(-1), partition_id(-1), ratio(0.0), partition_degree(0), total_degree(0) {}
    FrontierNode(int v, double r, int pd, int td, int pid = -1) : vertex(v), partition_id(pid), ratio(r), partition_degree(pd), total_degree(td) {}

    bool operator<(const FrontierNode &other) const {
        return ratio < other.ratio;
    }

    bool operator>(const FrontierNode &other) const {
        return ratio > other.ratio;
    }
};

struct PartitionUpdate {
    int partition_id;
    int node;

    PartitionUpdate() : partition_id(-1), node(-1) {}
    PartitionUpdate(int pid, int num) : partition_id(pid), node(num) {}
};

// 그래프 구조체 (CSR) 각 프로세스마다 로컬 그래프를 표현
// global_ids: 로컬 노드의 글로벌 ID 배열
// vertex_labels: 로컬 노드의 라벨 (소유 노드만 변경 가능)
// row_ptr: 각 정점의 시작 인덱스 (CSR 형식)
// col_indices: 인접 정점의 인덱스 배열 (로컬: 0~num_vertices-1, Ghost: num_vertices~)
// num_edges: 총 엣지 수
// num_vertices: 로컬 정점 수
struct Graph {
    int num_vertices = 0;
    int num_edges = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
    std::vector<int> global_ids; // 로컬 노드의 글로벌 ID 배열
    std::vector<int> vertex_labels; // 로컬 노드의 라벨

    void clear() {
        num_vertices = 0;
        num_edges = 0;
        row_ptr.clear();
        col_indices.clear();
        global_ids.clear();
        vertex_labels.clear();
    }
};

// Ghost 노드 정보 구조체 각 스레드마다 Ghost 노드 정보를 표현
// global_ids: Ghost 노드의 글로벌 ID 배열
// ghost_labels: Ghost 노드의 라벨 배열
// global_to_local: 글로벌 ID → 로컬 인덱스 매핑
struct GhostNodes {
    std::vector<int> global_ids;  // 글로벌 ID
    std::vector<int> ghost_labels; // Ghost 노드의 라벨
    std::unordered_map<int, int> global_to_local; // 글로벌 → 로컬 인덱스 매핑

    void clear() {
        global_ids.clear();
        ghost_labels.clear();
        global_to_local.clear();
    }
};

// 파티션별 통계 구조체
struct PartitionStats {
    std::vector<int> local_vertex_counts;
    std::vector<int> local_edge_counts;
    std::vector<int> global_vertex_counts;
    std::vector<int> global_edge_counts;
    int total_vertices;
    int total_edges;
    double expected_vertices;
    double expected_edges;
};

// Phase 1 메트릭 구조체 
struct Phase1Metrics {
    int initial_edge_cut = 0;
    double initial_vertex_balance = 0.0;
    double initial_edge_balance = 0.0;
    long loading_time_ms = 0;
    long distribution_time_ms = 0;
    int total_vertices = 0;
    int total_edges = 0;
    std::vector<int> partition_vertex_counts;
    std::vector<int> partition_edge_counts;
};

// Phase 1/2 공통 비교용 메트릭 구조체
struct PartitioningMetrics {
    int edge_cut = 0;
    double vertex_balance = 0.0;
    double edge_balance = 0.0;
    long loading_time_ms = 0;
    long distribution_time_ms = 0;
    int num_partitions = 0;
    int total_vertices = 0;
    int total_edges = 0;

    PartitioningMetrics() = default;
    PartitioningMetrics(const Phase1Metrics& m, int num_parts) {
        edge_cut = m.initial_edge_cut;
        vertex_balance = m.initial_vertex_balance;
        edge_balance = m.initial_edge_balance;
        loading_time_ms = m.loading_time_ms;
        distribution_time_ms = m.distribution_time_ms;
        num_partitions = num_parts;
        total_vertices = m.total_vertices;
        total_edges = m.total_edges;
    }
};
#endif // DMOLP_GRAPH_TYPES_H
