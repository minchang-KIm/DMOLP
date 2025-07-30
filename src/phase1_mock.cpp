#include <string>
#include <vector>
#include <fstream>
#include <mpi.h>
#include "types.h"

// --- 균등 분할 기반 mock phase1 ---
static int g_global_num_vertices = 0;
static int g_global_num_edges = 0;
int get_global_num_vertices() { return g_global_num_vertices; }
int get_global_num_edges() { return g_global_num_edges; }

// 파일에서 edge list를 읽어 CSR로 변환
static void load_graph_from_file(const std::string& filename, Graph& graph) {
    std::ifstream fin(filename);
    int max_v = 0, u, v;
    std::vector<std::pair<int, int>> edges;
    while (fin >> u >> v) {
        edges.emplace_back(u, v);
        max_v = std::max(max_v, std::max(u, v));
    }
    graph.num_vertices = max_v + 1;
    graph.num_edges = edges.size();
    g_global_num_vertices = graph.num_vertices;
    g_global_num_edges = graph.num_edges;
    graph.row_ptr.assign(graph.num_vertices + 1, 0);
    for (auto& e : edges) graph.row_ptr[e.first + 1]++;
    for (int i = 1; i <= graph.num_vertices; ++i)
        graph.row_ptr[i] += graph.row_ptr[i - 1];
    graph.col_indices.resize(edges.size());
    std::vector<int> cur(graph.num_vertices, 0);
    for (auto& e : edges) {
        int pos = graph.row_ptr[e.first] + cur[e.first];
        graph.col_indices[pos] = e.second;
        cur[e.first]++;
    }
}

// phase1 mock: 균등 분할, 각 rank는 자신 파티션만 보유
Phase1Metrics phase1_partition_and_distribute(int part_id, int num_partitions, int total_partitions, Graph& local_graph, std::vector<int>& vertex_labels, const std::string& filename) {
    Graph full_graph;
    load_graph_from_file(filename, full_graph);
    int n = full_graph.num_vertices;
    int per_part = (n + total_partitions - 1) / total_partitions;
    int start = part_id * per_part;
    int end = std::min(n, start + per_part);
    // local CSR 추출 (col_indices에 global id만 저장)
    local_graph.num_vertices = end - start;
    local_graph.row_ptr.assign(local_graph.num_vertices + 1, 0);
    local_graph.col_indices.clear();
    std::vector<int> mapping(n, -1);
    for (int i = start; i < end; ++i) mapping[i] = i - start;
    for (int u = start; u < end; ++u) {
        int local_u = mapping[u];
        for (int idx = full_graph.row_ptr[u]; idx < full_graph.row_ptr[u+1]; ++idx) {
            int v = full_graph.col_indices[idx];
            local_graph.col_indices.push_back(v); // 항상 global id 저장
            local_graph.row_ptr[local_u + 1]++;
        }
    }
    for (int i = 1; i <= local_graph.num_vertices; ++i)
        local_graph.row_ptr[i] += local_graph.row_ptr[i-1];
    local_graph.num_edges = local_graph.col_indices.size();
    // local vertex_labels: 모두 part_id로
    vertex_labels.assign(local_graph.num_vertices, part_id);
    // 경계 노드(BV) 추출 예시 (이웃 중 mapping[v]==-1인 v가 있으면 경계 노드)
    std::vector<int> boundary_vertices;
    for (int local_u = 0; local_u < local_graph.num_vertices; ++local_u) {
        int row_start = local_graph.row_ptr[local_u];
        int row_end = local_graph.row_ptr[local_u+1];
        for (int idx = row_start; idx < row_end; ++idx) {
            int v = local_graph.col_indices[idx];
            if (mapping[v] == -1) {
                boundary_vertices.push_back(local_u);
                break;
            }
        }
    }
    // 메트릭
    Phase1Metrics m = {};
    m.total_vertices = full_graph.num_vertices;
    m.total_edges = full_graph.num_edges;
    m.partition_vertex_counts.assign(num_partitions, 0);
    m.partition_edge_counts.assign(num_partitions, 0);
    for (int u = 0; u < n; ++u) {
        int p = u / per_part;
        if (p >= 0 && p < num_partitions) {
            m.partition_vertex_counts[p]++;
            m.partition_edge_counts[p] += full_graph.row_ptr[u+1] - full_graph.row_ptr[u];
        }
    }
    m.initial_edge_cut = 0;
    m.initial_vertex_balance = 0;
    m.initial_edge_balance = 0;
    m.loading_time_ms = 0;
    m.distribution_time_ms = 0;
    // boundary_vertices, remote_neighbors 등은 필요시 반환/출력/통신에 활용
    return m;
}
