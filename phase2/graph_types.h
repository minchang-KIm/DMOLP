#ifndef DMOLP_GRAPH_TYPES_H
#define DMOLP_GRAPH_TYPES_H

#include <vector>

// 그래프 구조체 (CSR)
struct Graph {
    int num_vertices = 0;
    int num_edges = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
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

// 파티션 정보 구조체 (Phase2 등에서 사용)
struct PartitionInfo {
    int partition_id = 0;
    double RV = 1.0;  // Ratio of Vertex
    double RE = 1.0;  // Ratio of Edge
    double imb_RV = 0.0;  // imbalance RV
    double imb_RE = 0.0;  // imbalance RE
    double G_RV = 0.0;    // Gain RV
    double G_RE = 0.0;    // Gain RE
    double P_L = 0.0;     // Penalty function
};

// 파티션 업데이트 배열들 (Phase2 등에서 사용)
struct PartitionUpdate {
    std::vector<int> PU_RO;  // 자신 파티션 추가 정점들
    std::vector<int> PU_OV;  // 다른 파티션으로 보낼 정점들 (adjacency list)
    std::vector<std::pair<int, int>> PU_ON;  // 다른 파티션으로 보낼 이웃 정보 (key-value)
    std::vector<int> PU_RV;  // 다른 파티션으로부터 받은 정점들 (adjacency list)
    std::vector<std::pair<int, int>> PU_RN;  // 다른 파티션으로부터 받은 이웃 정보 (key-value)
};

#endif // DMOLP_GRAPH_TYPES_H
