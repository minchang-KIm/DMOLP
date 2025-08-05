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

void merge_csr(int mpi_rank, const std::unordered_map<int, Graph> &tmp_graph, const std::unordered_map<int, GhostNodes> &tmp_ghost, Graph &local_graph, GhostNodes &ghost_nodes) {
    local_graph.clear();
    ghost_nodes.clear();

    std::vector<int> partition_ids;
    for (const auto &[partition_id, graph] : tmp_graph) {
        partition_ids.push_back(partition_id);
    }
    std::sort(partition_ids.begin(), partition_ids.end());

    for (int partition_id : partition_ids) {
        const auto &graph = tmp_graph.at(partition_id);
        local_graph.global_ids.insert(local_graph.global_ids.end(), graph.global_ids.begin(), graph.global_ids.end());
        local_graph.vertex_labels.insert(local_graph.vertex_labels.end(), graph.vertex_labels.begin(), graph.vertex_labels.end());
    }

    for (int partition_id : partition_ids) {
        if (tmp_ghost.find(partition_id) != tmp_ghost.end()) {
            const auto &ghost = tmp_ghost.at(partition_id);
            ghost_nodes.global_ids.insert(ghost_nodes.global_ids.end(), ghost.global_ids.begin(), ghost.global_ids.end());
            ghost_nodes.ghost_labels.insert(ghost_nodes.ghost_labels.end(), ghost.ghost_labels.begin(), ghost.ghost_labels.end());
        }
    }

    local_graph.num_vertices = static_cast<int>(local_graph.global_ids.size());

    for (size_t i = 0; i < ghost_nodes.global_ids.size(); i++) {
        ghost_nodes.global_to_local[ghost_nodes.global_ids[i]] = i;
    }

    std::unordered_map<int, int> local_global_to_local;
    for (size_t i = 0; i < local_graph.global_ids.size(); ++i) {
        local_global_to_local[local_graph.global_ids[i]] = i;
    }
    
    local_graph.row_ptr.resize(local_graph.num_vertices + 1, 0);
    int total_edges = 0;
    int current_vertex = 0;
    
    for (int partition_id : partition_ids) {
        const auto& graph = tmp_graph.at(partition_id);
        const auto& ghost = tmp_ghost.at(partition_id);
        
        for (int i = 0; i < graph.num_vertices; ++i) {
            for (int e = graph.row_ptr[i]; e < graph.row_ptr[i + 1]; ++e) {
                int neighbor_local = graph.col_indices[e];
                int new_neighbor_idx;
                
                if (neighbor_local < graph.num_vertices) {
                    int neighbor_gid = graph.global_ids[neighbor_local];
                    new_neighbor_idx = local_global_to_local[neighbor_gid];
                } else {
                    int ghost_idx = neighbor_local - graph.num_vertices;
                    int neighbor_gid = ghost.global_ids[ghost_idx];
                    new_neighbor_idx = local_graph.num_vertices + ghost_nodes.global_to_local[neighbor_gid];
                }
                
                local_graph.col_indices.push_back(new_neighbor_idx);
                total_edges++;
            }
            
            current_vertex++;
            local_graph.row_ptr[current_vertex] = total_edges;
        }
    }
    
    local_graph.num_edges = total_edges;
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

    std::unordered_map<int, Graph> tmp_graph;
    std::unordered_map<int, GhostNodes> tmp_ghost;
    partition_expansion(mpi_rank, mpi_size, num_parts, theta, seeds, global_degree, graph, partitions, tmp_graph, tmp_ghost);
    
    double distribute_end = MPI_Wtime();

    merge_csr(mpi_rank, tmp_graph, tmp_ghost, local_graph, ghost_nodes);

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