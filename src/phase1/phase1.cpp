#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "phase1/phase1.h"
#include "phase1/init.hpp"
#include "phase1/partition.hpp"
#include "phase1/seed.hpp"
#include "utils.hpp"
#include "graph_types.h"

void convert_to_csr(
    int mpi_rank,
    const std::unordered_map<int, std::vector<int>> &graph,
    const std::vector<std::vector<int>> &partitions,
    Graph &local_graph,
    GhostNodes &ghost_nodes
) {
    const std::vector<int> &owned_nodes = partitions[mpi_rank];
    std::unordered_map<int, int> global_to_local;

    int local_idx = 0;
    for (int gid : owned_nodes) {
        global_to_local[gid] = local_idx++;
    }

    local_graph.num_vertices = owned_nodes.size();
    local_graph.row_ptr.resize(owned_nodes.size() + 1, 0);
    local_graph.global_ids = owned_nodes;
    local_graph.vertex_labels.resize(owned_nodes.size(), mpi_rank);  // 초기 라벨은 파티션 번호로

    std::vector<int> &row_ptr = local_graph.row_ptr;
    std::vector<int> &col_indices = local_graph.col_indices;

    int edge_count = 0;
    for (size_t i = 0; i < owned_nodes.size(); ++i) {
        int gid = owned_nodes[i];
        const auto &neighbors = graph.at(gid);
        for (int ngid : neighbors) {
            if (global_to_local.count(ngid)) {
                // 로컬 정점
                col_indices.push_back(global_to_local[ngid]);
            } else {
                // Ghost 노드
                if (ghost_nodes.global_to_local.count(ngid) == 0) {
                    int ghost_idx = ghost_nodes.global_ids.size();
                    ghost_nodes.global_ids.push_back(ngid);
                    ghost_nodes.ghost_labels.push_back(-1); // 아직 라벨 미정
                    ghost_nodes.global_to_local[ngid] = ghost_idx;
                }
                int ghost_idx = ghost_nodes.global_to_local[ngid];
                col_indices.push_back(local_graph.num_vertices + ghost_idx); // 로컬 뒤쪽에 배치
            }
            edge_count++;
        }
        row_ptr[i + 1] = edge_count;
    }

    local_graph.num_edges = edge_count;
}
// === Phase1 메인 ===
Phase1Metrics run_phase1(
    int mpi_rank, int mpi_size,
    const char* graph_file,
    int num_parts,
    int theta,
    Graph &local_graph,
    GhostNodes &ghost_nodes
)
{
    Phase1Metrics metrics{};

    // 그래프 데이터 로드
    std::unordered_map<int, std::vector<int>> graph;
    std::unordered_map<int, int> local_degree;
    std::unordered_map<int, int> global_degree;
    std::vector<int> hub_nodes;
    std::vector<int> landmarks;
    std::vector<int> seeds;
    std::vector<std::vector<int>> partitions;
    int V;
    int E;

    double load_start = MPI_Wtime();

    load_graph(graph_file, mpi_rank, mpi_size, graph, local_degree, V, E);

    
    double distribute_start = MPI_Wtime();

    gather_degrees(local_degree, global_degree, mpi_rank, mpi_size);

    if (mpi_rank == 0) {
        hub_nodes = find_hub_nodes(global_degree);
        landmarks = find_landmarks(global_degree);
    }

    sync_vector(mpi_rank, 0, hub_nodes);
    sync_vector(mpi_rank, 0, landmarks);

    seeds = find_seeds(mpi_rank, mpi_size, num_parts, landmarks, hub_nodes, global_degree, graph);

    sync_vector(mpi_rank, 0, seeds);

    partition_expansion(mpi_rank, mpi_size, num_parts, theta, seeds, global_degree, graph, partitions);
    
    double distribute_end = MPI_Wtime();

    convert_to_csr(mpi_rank, graph, partitions, local_graph, ghost_nodes);

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
    metrics.partition_vertex_counts.assign(num_parts, 0);
    for (int i = 0; i < (int)local_graph.global_ids.size(); i++)
        metrics.partition_vertex_counts[local_graph.vertex_labels[i]]++;
    MPI_Allreduce(MPI_IN_PLACE, metrics.partition_vertex_counts.data(), num_parts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === 파티션별 edge count ===
    metrics.partition_edge_counts.assign(num_parts, 0);
    for (int u = 0; u < (int)local_graph.global_ids.size(); u++) {
        int u_label = local_graph.vertex_labels[u];
        for (int e = local_graph.row_ptr[u]; e < local_graph.row_ptr[u + 1]; e++) {
            int v_local = local_graph.col_indices[e];
            if (local_graph.vertex_labels[v_local] == u_label)
                metrics.partition_edge_counts[u_label]++;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, metrics.partition_edge_counts.data(), num_parts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === Balance 계산 ===
    double expected_vertices = static_cast<double>(V) / num_parts;
    double max_vertex_ratio = 0;
    for (int count : metrics.partition_vertex_counts)
        max_vertex_ratio = std::max(max_vertex_ratio, count / expected_vertices);
    metrics.initial_vertex_balance = max_vertex_ratio;

    double total_partition_edges = 0;
    for (int count : metrics.partition_edge_counts) total_partition_edges += count;
    double expected_edges = total_partition_edges / num_parts;
    double max_edge_ratio = 0;
    for (int count : metrics.partition_edge_counts)
        max_edge_ratio = std::max(max_edge_ratio, count / expected_edges);
    metrics.initial_edge_balance = max_edge_ratio;

    metrics.loading_time_ms = (distribute_start - load_start) * 1000.0;
    metrics.distribution_time_ms = (distribute_end - distribute_start) * 1000.0;
    metrics.total_vertices = V;
    metrics.total_edges = E;

    return metrics;

}