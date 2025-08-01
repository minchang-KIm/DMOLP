#include "phase1.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <set>

static bool loadGraphInfo(const std::string& filename, int& V, int& E) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    file >> V >> E;
    return true;
}

static void readMyVertices(
    const std::string& filename,
    const std::vector<int>& my_vertices,
    std::vector<std::vector<int>>& my_adj)
{
    std::ifstream file(filename);
    int V, E;
    file >> V >> E;
    std::string line;
    std::getline(file, line); // 첫 줄 버림

    // 빠른 검색을 위해 set으로 소유 vertex 관리
    std::set<int> owned(my_vertices.begin(), my_vertices.end());

    int v = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (owned.count(v)) {
            std::istringstream iss(line);
            int nb;
            while (iss >> nb) {
                my_adj[v].push_back(nb - 1);
            }
        }
        v++;
    }
}

Phase1Metrics phase1_partition_and_distribute(
    int mpi_rank, int mpi_size, int num_partitions,
    const std::string& filename,
    Graph& local_graph,
    std::vector<int>& vertex_labels,
    std::vector<int>& global_ids,
    std::unordered_map<int,int>& global_to_local
) {
    Phase1Metrics metrics{};

    auto t0 = std::chrono::high_resolution_clock::now();

    // === 그래프 크기 (모두 동일하게 접근) ===
    int V = 0, E = 0;
    loadGraphInfo(filename, V, E);

    auto t_load_end = std::chrono::high_resolution_clock::now();

    // === 라벨 = 파티션 ID ===
    std::vector<int> global_labels(V, -1);
    int vertices_per_partition = V / num_partitions;
    for (int v = 0; v < V; v++) {
        int pid = std::min(v / vertices_per_partition, num_partitions - 1);
        global_labels[v] = pid;
    }

    auto partition_to_rank = [&](int pid) { return pid % mpi_size; };

    // === 내 Rank 담당 vertex 집합 ===
    std::vector<int> owned_vertices;
    for (int v = 0; v < V; v++) {
        int pid = global_labels[v];
        if (partition_to_rank(pid) == mpi_rank) {
            owned_vertices.push_back(v);
        }
    }

    // === 내 vertex adjacency 읽기 (분산 I/O) ===
    std::vector<std::vector<int>> my_adj(V); // V 크기이지만 owned만 채움
    readMyVertices(filename, owned_vertices, my_adj);

    // === ghost node 포함 vertex 집합 ===
    std::set<int> ghost_nodes;
    for (int g : owned_vertices) {
        for (int nb : my_adj[g]) {
            if (std::find(owned_vertices.begin(), owned_vertices.end(), nb) == owned_vertices.end()) {
                ghost_nodes.insert(nb);
            }
        }
    }

    // === global → local index 매핑 ===
    global_to_local.clear();
    global_ids.clear();
    int local_idx = 0;
    for (int g : owned_vertices) {
        global_to_local[g] = local_idx++;
        global_ids.push_back(g);
    }
    for (int g : ghost_nodes) {
        global_to_local[g] = local_idx++;
        global_ids.push_back(g);
    }

    // === CSR 구성 ===
    local_graph.num_vertices = global_ids.size();
    vertex_labels.resize(local_graph.num_vertices);
    local_graph.row_ptr.resize(local_graph.num_vertices + 1, 0);

    int edge_cnt = 0;
    std::vector<int> col_indices;
    for (int i = 0; i < (int)owned_vertices.size(); i++) {
        int g = owned_vertices[i];
        vertex_labels[i] = global_labels[g];
        for (int nb : my_adj[g]) {
            col_indices.push_back(global_to_local[nb]);
            edge_cnt++;
        }
        local_graph.row_ptr[i+1] = edge_cnt;
    }
    // ghost node 처리 (edge 없음)
    for (int gi = owned_vertices.size(); gi < (int)global_ids.size(); gi++) {
        vertex_labels[gi] = global_labels[global_ids[gi]];
        local_graph.row_ptr[gi+1] = edge_cnt;
    }
    local_graph.col_indices = std::move(col_indices);
    local_graph.num_edges = edge_cnt;

    auto t1 = std::chrono::high_resolution_clock::now();

    // === Edge-cut 계산 ===
    int local_edge_cut = 0;
    for (int u = 0; u < (int)owned_vertices.size(); u++) {
        int u_label = vertex_labels[u];
        for (int e = local_graph.row_ptr[u]; e < local_graph.row_ptr[u+1]; e++) {
            int v_local = local_graph.col_indices[e];
            int v_label = vertex_labels[v_local];
            if (u_label != v_label) local_edge_cut++;
        }
    }
    MPI_Allreduce(&local_edge_cut, &metrics.initial_edge_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === 파티션별 vertex count ===
    metrics.partition_vertex_counts.assign(num_partitions, 0);
    for (int i = 0; i < (int)owned_vertices.size(); i++)
        metrics.partition_vertex_counts[vertex_labels[i]]++;
    MPI_Allreduce(MPI_IN_PLACE, metrics.partition_vertex_counts.data(), num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === 파티션별 edge count ===
    metrics.partition_edge_counts.assign(num_partitions, 0);
    for (int u = 0; u < (int)owned_vertices.size(); u++) {
        int u_label = vertex_labels[u];
        for (int e = local_graph.row_ptr[u]; e < local_graph.row_ptr[u+1]; e++) {
            int v_local = local_graph.col_indices[e];
            if (vertex_labels[v_local] == u_label)
                metrics.partition_edge_counts[u_label]++;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, metrics.partition_edge_counts.data(), num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === Balance 계산 ===
    double expected_vertices = static_cast<double>(V) / num_partitions;
    double max_vertex_ratio = 0;
    for (int count : metrics.partition_vertex_counts)
        max_vertex_ratio = std::max(max_vertex_ratio, count / expected_vertices);
    metrics.initial_vertex_balance = max_vertex_ratio;

    double total_partition_edges = 0;
    for (int count : metrics.partition_edge_counts) total_partition_edges += count;
    double expected_edges = total_partition_edges / num_partitions;
    double max_edge_ratio = 0;
    for (int count : metrics.partition_edge_counts)
        max_edge_ratio = std::max(max_edge_ratio, count / expected_edges);
    metrics.initial_edge_balance = max_edge_ratio;

    metrics.loading_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_load_end - t0).count();
    metrics.distribution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t_load_end).count();
    metrics.total_vertices = V;
    metrics.total_edges = E;

    return metrics;
}