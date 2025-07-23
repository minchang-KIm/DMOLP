#ifndef DMOLP_TYPES_H
#define DMOLP_TYPES_H

#include <vector>
#include <unordered_map>

// 파티션 정보 구조체 (PI)
struct PartitionInfo {
    int partition_id;
    double RV;  // Ratio of Vertex
    double RE;  // Ratio of Edge
    double imb_RV;  // imbalance RV
    double imb_RE;  // imbalance RE
    double G_RV;    // Gain RV
    double G_RE;    // Gain RE
    double P_L;     // Penalty function
};

// 파티션 업데이트 배열들
struct PartitionUpdate {
    std::vector<int> PU_RO;  // 자신 파티션 추가 정점들
    std::vector<int> PU_OV;  // 다른 파티션으로 보낼 정점들 (adjacency list)
    std::vector<std::pair<int, int>> PU_ON;  // 다른 파티션으로 보낼 이웃 정보 (key-value)
    std::vector<int> PU_RV;  // 다른 파티션으로부터 받은 정점들 (adjacency list)
    std::vector<std::pair<int, int>> PU_RN;  // 다른 파티션으로부터 받은 이웃 정보 (key-value)
};

// 그래프 구조체 (phase1.cu에서 정의된 것과 동일)
struct Graph {
    int num_vertices;
    int num_edges;
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
};

// Phase 1 메트릭 구조체 (phase1.cu에서 정의된 것과 동일)
struct Phase1Metrics {
    int total_vertices;
    int total_edges;
    int initial_edge_cut;
    double initial_vertex_balance;
    double initial_edge_balance;
    long loading_time_ms;
    long distribution_time_ms;
    std::vector<int> partition_vertex_counts;
    std::vector<int> partition_edge_counts;
};

// 수렴 조건 상수
constexpr double EPSILON = 0.005;  // 더 엄격한 수렴 조건
constexpr int MAX_CONVERGENCE_COUNT = 5;  // 연속 수렴 횟수

#endif // DMOLP_TYPES_H
