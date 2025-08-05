#ifndef DMOLP_GRAPH_TYPES_H
#define DMOLP_GRAPH_TYPES_H

#include <vector>
#include <unordered_map>
#include <unordered_set>

struct BFSResult {
    std::vector<std::unordered_set<int>> levels;
    std::unordered_set<int> all_visited;

    void clear() {
        levels.clear();
        all_visited.clear();
    }

    void ensure_level(int level) {
        if (level >= static_cast<int>(levels.size())) levels.resize(level + 1);
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

// 그래프 구조체 (CSR) 각 스레드마다 로컬 그래프를 표현
// global_ids: 글로벌 ID 배열
// vertex_labels: 각 정점의 라벨
// row_ptr: 각 정점의 시작 인덱스 (CSR 형식)
// col_indices: 인접 정점의 로컬 인덱스 배열
// num_edges: 총 엣지 수
// num_vertices: 총 정점 수
// vertex_labels: 각 정점의 라벨
struct Graph {
    int num_vertices = 0;
    int num_edges = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
    std::vector<int> global_ids; // 글로벌 ID 배열
    std::vector<int> vertex_labels; // 각 정점의 라벨
};

// Ghost 노드 정보 구조체 각 스레드마다 Ghost 노드 정보를 표현
// global_ids: Ghost 노드의 글로벌 ID 배열
// ghost_labels: Ghost 노드의 라벨 배열
// global_to_local: 글로벌 ID → 로컬 인덱스 매핑
struct GhostNodes {
    std::vector<int> global_ids;  // 글로벌 ID
    std::vector<int> ghost_labels; // Ghost 노드의 라벨
    std::unordered_map<int, int> global_to_local; // 글로벌 → 로컬 인덱스 매핑
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

    PartitioningMetrics() = default;
    PartitioningMetrics(const Phase1Metrics& m, int num_parts) {
        edge_cut = m.initial_edge_cut;
        vertex_balance = m.initial_vertex_balance;
        edge_balance = m.initial_edge_balance;
        loading_time_ms = m.loading_time_ms;
        distribution_time_ms = m.distribution_time_ms;
        num_partitions = num_parts;
    }
};


// 파티션 정보 구조체 (Phase2 등에서 사용)
struct PartitionInfo {
    double RV = 1.0;  // Ratio of Vertex
    double RE = 1.0;  // Ratio of Edge
    double imb_RV = 0.0;  // imbalance RV
    double imb_RE = 0.0;  // imbalance RE
    double G_RV = 0.0;    // Gain RV
    double G_RE = 0.0;    // Gain RE
    double P_L = 0.0;     // Penalty function
};


#endif // DMOLP_GRAPH_TYPES_H
