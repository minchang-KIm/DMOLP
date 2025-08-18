#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <chrono>

#include "phase1/phase1.h"
#include "phase1/init.hpp"
#include "phase1/partition.hpp"
#include "phase1/seed.hpp"
#include "utils.hpp"
#include "graph_types.h"

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
    std::pair<int, int> first_seed;
    std::vector<int> seeds;
    std::vector<std::vector<int>> partitions;
    int V;
    uint64_t E;

    auto load_start = std::chrono::high_resolution_clock::now();

    load_graph(graph_file, mpi_rank, mpi_size, graph, local_degree, V, E);
    
    auto distribute_start = std::chrono::high_resolution_clock::now();

    gather_degrees(local_degree, global_degree, mpi_rank, mpi_size);

    if (mpi_rank == 0) {
        hub_nodes = find_hub_nodes(global_degree);
    }

    first_seed = find_first_seed(global_degree);

    sync_vector(mpi_rank, 0, hub_nodes);

    seeds = find_seeds(mpi_rank, mpi_size, num_parts, V, first_seed, hub_nodes, global_degree, graph);

    sync_vector(mpi_rank, 0, seeds);

    partition_expansion(mpi_rank, mpi_size, num_parts, theta, seeds, global_degree, graph, local_graph, ghost_nodes, false);
    
    auto distribute_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(distribute_start - load_start);
    auto distribution_duration = std::chrono::duration_cast<std::chrono::milliseconds>(distribute_end - distribute_start);

    // Ghost 노드 라벨 안전하게 가져오는 헬퍼 함수 (Phase2와 동일)
    auto getNodeLabel = [&](int node_id) -> int {
        if (node_id < local_graph.num_vertices) {
            return local_graph.vertex_labels[node_id];
        } else {
            int ghost_idx = node_id - local_graph.num_vertices;
            return (ghost_idx >= 0 && ghost_idx < (int)ghost_nodes.ghost_labels.size()) 
                   ? ghost_nodes.ghost_labels[ghost_idx] : -1;
        }
    };

    // Edge-cut 계산 (Phase2와 동일한 방식: owned 노드만 카운트하여 중복 방지)
    int local_edge_cut = 0;
    for (int u = 0; u < local_graph.num_vertices; u++) {  // owned 노드만
        int u_label = local_graph.vertex_labels[u];
        for (int e = local_graph.row_ptr[u]; e < local_graph.row_ptr[u + 1]; e++) {
            int v = local_graph.col_indices[e];
            
            // v의 라벨 결정 (ghost 노드 처리)
            int v_label = getNodeLabel(v);
            
            // 다른 파티션 간 간선이면 edge-cut에 포함
            if (u_label != -1 && v_label != -1 && u_label != v_label) {
                local_edge_cut++;
            }
        }
    }
    MPI_Allreduce(&local_edge_cut, &metrics.initial_edge_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    metrics.initial_edge_cut /= 2; // undirected 그래프이므로 edge-cut은 2배로 계산됨

    // === 파티션별 vertex count ===
    metrics.partition_vertex_counts.assign(num_parts, 0);
    for (int i = 0; i < (int)local_graph.global_ids.size(); i++){
        if (local_graph.vertex_labels[i] >= 0){
            metrics.partition_vertex_counts[local_graph.vertex_labels[i]]++;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, metrics.partition_vertex_counts.data(), num_parts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // === 파티션별 edge count ===
    std::vector<int> local_partition_edges(num_parts, 0);

    // 3. 로컬 노드만 순회하며 파티션 내부 간선을 카운트합니다.
    for (int u = 0; u < local_graph.num_vertices; u++) {
        int u_label = local_graph.vertex_labels[u];
        // 할당되지 않은 노드는 건너뜁니다.
        if (u_label < 0) continue;

        for (int e = local_graph.row_ptr[u]; e < local_graph.row_ptr[u + 1]; e++) {
            int v_local = local_graph.col_indices[e];
            
            // 위에서 정의한 getNodeLabel 람다 함수를 사용하여 안전하게 라벨을 조회합니다.
            if (getNodeLabel(v_local) == u_label) {
                local_partition_edges[u_label]++;
            }
        }
    }

    metrics.partition_edge_counts.resize(num_parts);

    // 4. MPI_Allreduce를 사용해 모든 랭크의 로컬 카운트를 합산하여 최종 결과를 얻습니다.
    MPI_Allreduce(local_partition_edges.data(), metrics.partition_edge_counts.data(), num_parts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // 5. 무방향 그래프에서 각 간선이 두 번씩 카운트되었으므로, 2로 나누어 보정합니다.
    for (int i = 0; i < num_parts; i++) {
        metrics.partition_edge_counts[i] /= 2;
    }


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

    metrics.loading_time_ms = load_duration.count();
    metrics.distribution_time_ms = distribution_duration.count();
    metrics.total_vertices = V;
    metrics.total_edges = E;

    std::cout << "[Rank " << mpi_rank << "] Phase1 함수 완료 준비" << std::endl;
    std::cout.flush();

    return metrics;

}