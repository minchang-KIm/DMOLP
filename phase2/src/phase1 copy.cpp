#include "phase1.h"
#include "graph_types.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <set>

// === 그래프 크기 로드 ===
static bool loadGraphInfo(const std::string& filename, int& V, int& E) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    file >> V >> E;
    return true;
}

// === 내 Rank가 소유한 노드 인접리스트 읽기 ===
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

    std::set<int> owned(my_vertices.begin(), my_vertices.end());

    int v = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (owned.count(v)) {
            std::istringstream iss(line);
            int nb;
            while (iss >> nb) {
                my_adj[v].push_back(nb - 1); // 파일이 1-based라고 가정
            }
        }
        v++;
    }
}

// === Phase1 메인 ===
Phase1Metrics phase1_partition_and_distribute(
    int mpi_rank, int mpi_size, int num_partitions,
    const std::string& filename,
    Graph& local_graph,
    GhostNodes& ghost_nodes,
    double thetta)
{
    Phase1Metrics metrics{};
    auto t0 = std::chrono::high_resolution_clock::now();

    // === 그래프 크기 ===
    int V = 0, E = 0;
    loadGraphInfo(filename, V, E);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0)
        std::cout << "[Phase1] 그래프 크기 로딩: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_load_end - t0).count()
                  << " ms\n";

    // === 초기 라벨 = 파티션 ID ===
    auto t_label_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_labels(V, -1);
    int vertices_per_partition = V / num_partitions;
    for (int v = 0; v < V; v++) {
        int pid = std::min(v / vertices_per_partition, num_partitions - 1);
        global_labels[v] = pid;
    }
    auto t_label_end = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0)
        std::cout << "[Phase1] 라벨링: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_label_end - t_label_start).count()
                  << " ms\n";

    auto partition_to_rank = [&](int pid) { return pid % mpi_size; };

    // === 내 Rank 담당 vertex 집합 ===
    auto t_owned_start = std::chrono::high_resolution_clock::now();
    std::vector<int> owned_vertices;
    for (int v = 0; v < V; v++) {
        int pid = global_labels[v];
        if (partition_to_rank(pid) == mpi_rank)
            owned_vertices.push_back(v);
    }
    auto t_owned_end = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0)
        std::cout << "[Phase1] 내 Rank 담당 vertex 집합 추출: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_owned_end - t_owned_start).count()
                  << " ms\n";

    // === 내 vertex adjacency 읽기 ===
    auto t_adj_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> my_adj(V);
    readMyVertices(filename, owned_vertices, my_adj);
    auto t_adj_end = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0)
        std::cout << "[Phase1] 내 vertex adjacency 읽기: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_adj_end - t_adj_start).count()
                  << " ms\n";

    // === Ghost node 집합 ===
    auto t_ghost_start = std::chrono::high_resolution_clock::now();
    std::set<int> ghost_set;
    for (int g : owned_vertices) {
        for (int nb : my_adj[g]) {
            if (std::find(owned_vertices.begin(), owned_vertices.end(), nb) == owned_vertices.end()) {
                ghost_set.insert(nb);
            }
        }
    }
    auto t_ghost_end = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0)
        std::cout << "[Phase1] Ghost node 집합 추출: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_ghost_end - t_ghost_start).count()
                  << " ms\n";

    // === local index ↔ global id (owned + ghost) ===
    auto t_map_start = std::chrono::high_resolution_clock::now();
    local_graph.global_ids.clear();
    ghost_nodes.global_ids.clear();
    ghost_nodes.global_to_local.clear();

    // owned 먼저
    for (int g : owned_vertices)
        local_graph.global_ids.push_back(g);

    // ghost 추가
    for (int g : ghost_set) {
        ghost_nodes.global_to_local[g] = (int)local_graph.global_ids.size() + (int)ghost_nodes.global_ids.size();
        ghost_nodes.global_ids.push_back(g);
    }
    auto t_map_end = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0)
        std::cout << "[Phase1] 매핑 구성: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_map_end - t_map_start).count()
                  << " ms\n";

    // === CSR 구성 ===
    auto t_csr_start = std::chrono::high_resolution_clock::now();
    int local_vertices_count = (int)local_graph.global_ids.size();
    int ghost_count = (int)ghost_nodes.global_ids.size();
    local_graph.num_vertices = local_vertices_count + ghost_count;

    local_graph.vertex_labels.resize(local_graph.num_vertices);
    local_graph.row_ptr.assign(local_graph.num_vertices + 1, 0);

    int edge_cnt = 0;
    std::vector<int> col_indices;

    // owned vertices
    for (int li = 0; li < (int)local_graph.global_ids.size(); li++) {
        int g = local_graph.global_ids[li];
        local_graph.vertex_labels[li] = global_labels[g];
        for (int nb : my_adj[g]) {
            int v_local = -1;
            // 내 로컬에 있으면 index 찾기
            auto it_owned = std::find(local_graph.global_ids.begin(), local_graph.global_ids.end(), nb);
            if (it_owned != local_graph.global_ids.end())
                v_local = (int)(it_owned - local_graph.global_ids.begin());
            else
                v_local = ghost_nodes.global_to_local[nb];
            col_indices.push_back(v_local);
            edge_cnt++;
        }
        local_graph.row_ptr[li + 1] = edge_cnt;
    }

    // ghost vertices (edge 없음)
    for (int gi = 0; gi < ghost_count; gi++) {
        int global_id = ghost_nodes.global_ids[gi];
        ghost_nodes.ghost_labels.push_back(global_labels[global_id]);
        local_graph.vertex_labels[local_vertices_count + gi] = global_labels[global_id];
        local_graph.row_ptr[local_vertices_count + gi + 1] = edge_cnt;
    }

    local_graph.col_indices = std::move(col_indices);
    local_graph.num_edges = edge_cnt;
    auto t_csr_end = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0)
        std::cout << "[Phase1] CSR 구성: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_csr_end - t_csr_start).count()
                  << " ms\n";

    // === Edge-cut 계산 ===
    int local_edge_cut = 0;
    for (int u = 0; u < (int)local_graph.global_ids.size(); u++) {
        int u_label = local_graph.vertex_labels[u];
        for (int e = local_graph.row_ptr[u]; e < local_graph.row_ptr[u + 1]; e++) {
            int v_local = local_graph.col_indices[e];
            int v_label = local_graph.vertex_labels[v_local];
            if (u_label != v_label) local_edge_cut++;
        }
    }
    MPI_Allreduce(&local_edge_cut, &metrics.initial_edge_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === 파티션별 vertex count ===
    metrics.partition_vertex_counts.assign(num_partitions, 0);
    for (int i = 0; i < (int)local_graph.global_ids.size(); i++)
        metrics.partition_vertex_counts[local_graph.vertex_labels[i]]++;
    MPI_Allreduce(MPI_IN_PLACE, metrics.partition_vertex_counts.data(), num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === 파티션별 edge count ===
    metrics.partition_edge_counts.assign(num_partitions, 0);
    for (int u = 0; u < (int)local_graph.global_ids.size(); u++) {
        int u_label = local_graph.vertex_labels[u];
        for (int e = local_graph.row_ptr[u]; e < local_graph.row_ptr[u + 1]; e++) {
            int v_local = local_graph.col_indices[e];
            if (local_graph.vertex_labels[v_local] == u_label)
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
    metrics.distribution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_csr_end - t_load_end).count();
    metrics.total_vertices = V;
    metrics.total_edges = E;

    return metrics;
}