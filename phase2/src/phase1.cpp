#include "phase1.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <algorithm>

// 간단한 그래프 로더 (METIS 형식)
static bool loadGraphFromFile(const std::string& filename, Graph& graph) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    int V, E;
    file >> V >> E;
    graph.num_vertices = V;
    graph.row_ptr.resize(V + 1, 0);
    std::vector<std::vector<int>> adj(V);

    int u = 0;
    std::string line;
    std::getline(file, line); // 첫줄 버림
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int neighbor;
        while (iss >> neighbor) {
            adj[u].push_back(neighbor - 1);
        }
        u++;
    }

    int edge_cnt = 0;
    for (int i = 0; i < V; i++) {
        graph.row_ptr[i] = edge_cnt;
        for (int nb : adj[i]) edge_cnt++;
    }
    graph.row_ptr[V] = edge_cnt;
    graph.col_indices.resize(edge_cnt);
    int idx = 0;
    for (int i = 0; i < V; i++) {
        for (int nb : adj[i]) graph.col_indices[idx++] = nb;
    }
    graph.num_edges = edge_cnt;
    return true;
}

#include <mpi.h>
#include <chrono>
#include <unordered_map>
#include "graph_types.h"
#include "phase1.h"

Phase1Metrics phase1_partition_and_distribute(
    int mpi_rank, int mpi_size, int num_partitions,
    const std::string& filename,
    Graph& local_graph,
    std::vector<int>& vertex_labels,
    std::vector<int>& global_ids,
    std::unordered_map<int,int>& global_to_local
) {
    Phase1Metrics metrics{};
    Graph full_graph;

    auto t0 = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0) {
        loadGraphFromFile(filename, full_graph);
    }

    // === 그래프 크기 정보 브로드캐스트 ===
    int graph_info[2] = {0, 0};
    if (mpi_rank == 0) {
        graph_info[0] = full_graph.num_vertices;
        graph_info[1] = full_graph.num_edges;
    }
    MPI_Bcast(graph_info, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int V = graph_info[0];
    int E = graph_info[1];
    int vertices_per_rank = V / mpi_size;
    int start_v = mpi_rank * vertices_per_rank;
    int end_v = (mpi_rank == mpi_size - 1) ? V : (mpi_rank + 1) * vertices_per_rank;

    auto t_load_end = std::chrono::high_resolution_clock::now();

    // === Rank0 → 다른 rank로 데이터 분배 ===
    if (mpi_rank == 0) {
        for (int rank = 1; rank < mpi_size; rank++) {
            int rs = rank * vertices_per_rank;
            int re = (rank == mpi_size - 1) ? V : (rank + 1) * vertices_per_rank;
            int rv = re - rs;

            std::vector<int> row_ptr(rv + 1);
            for (int i = 0; i <= rv; i++)
                row_ptr[i] = full_graph.row_ptr[rs + i] - full_graph.row_ptr[rs];
            int edges = full_graph.row_ptr[re] - full_graph.row_ptr[rs];

            MPI_Send(row_ptr.data(), rv + 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
            MPI_Send(&full_graph.col_indices[full_graph.row_ptr[rs]], edges, MPI_INT, rank, 1, MPI_COMM_WORLD);
        }
        // === 자기 데이터 복사 ===
        int lv = end_v - start_v;
        int edges = full_graph.row_ptr[end_v] - full_graph.row_ptr[start_v];
        local_graph.num_vertices = lv;
        local_graph.num_edges = edges;
        local_graph.row_ptr.resize(lv + 1);
        local_graph.col_indices.resize(edges);
        for (int i = 0; i <= lv; i++)
            local_graph.row_ptr[i] = full_graph.row_ptr[start_v + i] - full_graph.row_ptr[start_v];
        for (int i = 0; i < edges; i++)
            local_graph.col_indices[i] = full_graph.col_indices[full_graph.row_ptr[start_v] + i];
    } else {
        // === 데이터 수신 ===
        int lv = end_v - start_v;
        local_graph.num_vertices = lv;
        local_graph.row_ptr.resize(lv + 1);
        MPI_Recv(local_graph.row_ptr.data(), lv + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int edges = local_graph.row_ptr[lv];
        local_graph.num_edges = edges;
        local_graph.col_indices.resize(edges);
        MPI_Recv(local_graph.col_indices.data(), edges, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // === 라벨 초기화 (label = partition id) ===
    vertex_labels.resize(local_graph.num_vertices);
    int partitions_per_rank = num_partitions / mpi_size;
    int start_partition = mpi_rank * partitions_per_rank;
    for (int i = 0; i < local_graph.num_vertices; i++) {
        int global_id = start_v + i;
        vertex_labels[i] = start_partition + (i % partitions_per_rank);
        global_ids.push_back(global_id);
        global_to_local[global_id] = i;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // === 초기 edge-cut 계산 ===
    int local_edge_cut = 0;
    for (int u = 0; u < local_graph.num_vertices; u++) {
        int u_label = vertex_labels[u];
        for (int e = local_graph.row_ptr[u]; e < local_graph.row_ptr[u + 1]; e++) {
            int v = local_graph.col_indices[e];
            if (v < local_graph.num_vertices && u < v) {
                int v_label = vertex_labels[v];
                if (u_label != v_label) local_edge_cut++;
            }
        }
    }
    MPI_Allreduce(&local_edge_cut, &metrics.initial_edge_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === 파티션별 vertex count ===
    metrics.partition_vertex_counts.assign(num_partitions, 0);
    for (int i = 0; i < local_graph.num_vertices; i++)
        metrics.partition_vertex_counts[vertex_labels[i]]++;
    MPI_Allreduce(MPI_IN_PLACE, metrics.partition_vertex_counts.data(), num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === 파티션별 edge count ===
    metrics.partition_edge_counts.assign(num_partitions, 0);
    for (int u = 0; u < local_graph.num_vertices; u++) {
        int u_label = vertex_labels[u];
        for (int e = local_graph.row_ptr[u]; e < local_graph.row_ptr[u + 1]; e++) {
            int v = local_graph.col_indices[e];
            if (v < local_graph.num_vertices && vertex_labels[v] == u_label)
                metrics.partition_edge_counts[u_label]++;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, metrics.partition_edge_counts.data(), num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === balance 계산 ===
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

    // === 시간 기록 ===
    metrics.loading_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_load_end - t0).count();
    metrics.distribution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t_load_end).count();
    metrics.total_vertices = V;
    metrics.total_edges = E;

    return metrics;
}