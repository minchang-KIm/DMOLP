/**
 * @file phase2.cpp
 * @brief DMOLP Phase2 분산 그래프 파티셔닝 최적화 구현체 (동기화/통신 수정 버전)
 *
 * 변경 요약:
 * - Allgather 시점을 GPU 결과(delta_changes) 수집 "이후"로 이동
 * - std::async 제거, 동기 Allgatherv 사용 (또는 필요 시 MPI_Iallgatherv로 대체 가능)
 * - recv_deltas 적용 후 MPI_Barrier로 전역 스냅샷 일치 보장, 그 다음 통계/edge-cut 계산
 * - edge-cut 홀수 검사 조건 가독성 향상 및 MPI_Abort 사용
 * - Delta용 MPI datatype을 Type_create_struct로 안전하게 정의
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <omp.h>
#include <cstddef> // offsetof

#include "graph_types.h"
#include "phase2/phase2.h"
#include "phase2/gpu_lp_boundary.h"

// ==================== 유틸리티 함수들 ====================

/**
 * @brief Ghost 노드 라벨을 안전하게 조회하는 인라인 함수
 */
inline int getNodeLabel(int node_id, const Graph &g, const std::vector<int> &labels,
                        const GhostNodes &ghost_nodes) {
    if (node_id < g.num_vertices) {
        return labels[node_id];
    } else {
        int ghost_idx = node_id - g.num_vertices;
        if (ghost_idx < 0 || ghost_idx >= (int)ghost_nodes.ghost_labels.size()) {
            printf("Warning: Invalid ghost node index %d for original node %d\n", ghost_idx, node_id);
            return -1;
        }
        return ghost_nodes.ghost_labels[ghost_idx];
    }
}

/**
 * @brief 파티션별 통계를 계산하는 최적화된 함수
 */
static PartitionStats computePartitionStats(const Graph &g, const std::vector<int> &labels,
                                           const GhostNodes &ghost_nodes, int num_partitions) {
    PartitionStats stats;
    stats.local_vertex_counts.resize(num_partitions, 0);
    stats.local_edge_counts.resize(num_partitions, 0);
    stats.global_vertex_counts.resize(num_partitions, 0);
    stats.global_edge_counts.resize(num_partitions, 0);

    // 병렬 노드 카운팅
    #pragma omp parallel
    {
        std::vector<int> thread_vertex_counts(num_partitions, 0);
        #pragma omp for nowait schedule(static)
        for (int u = 0; u < g.num_vertices; u++) {
            int label = labels[u];
            if (label >= 0 && label < num_partitions) thread_vertex_counts[label]++;
        }
        #pragma omp critical
        {
            for (int i = 0; i < num_partitions; i++) stats.local_vertex_counts[i] += thread_vertex_counts[i];
        }
    }

    // 병렬 간선 카운팅 - 파티션 내부 간선만 계산
    #pragma omp parallel
    {
        std::vector<int> thread_edge_counts(num_partitions, 0);
        #pragma omp for nowait schedule(static)
        for (int u = 0; u < g.num_vertices; u++) {
            int label_u = labels[u];
            if (label_u < 0 || label_u >= num_partitions) continue;
            for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; e++) {
                int v = g.col_indices[e];
                int label_v = getNodeLabel(v, g, labels, ghost_nodes);
                if (label_v >= 0 && label_v < num_partitions && label_u == label_v) {
                    thread_edge_counts[label_u]++;
                }
            }
        }
        #pragma omp critical
        {
            for (int i = 0; i < num_partitions; i++) stats.local_edge_counts[i] += thread_edge_counts[i];
        }
    }

    // 단일 Allreduce로 집계
    std::vector<int> send_buffer(2 * num_partitions);
    std::vector<int> recv_buffer(2 * num_partitions);
    std::copy(stats.local_vertex_counts.begin(), stats.local_vertex_counts.end(), send_buffer.begin());
    std::copy(stats.local_edge_counts.begin(), stats.local_edge_counts.end(), send_buffer.begin() + num_partitions);
    MPI_Allreduce(send_buffer.data(), recv_buffer.data(), 2 * num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::copy(recv_buffer.begin(), recv_buffer.begin() + num_partitions, stats.global_vertex_counts.begin());
    std::copy(recv_buffer.begin() + num_partitions, recv_buffer.end(), stats.global_edge_counts.begin());

    // undirected 그래프 보정
    for (int i = 0; i < num_partitions; i++) stats.global_edge_counts[i] /= 2;

    stats.total_vertices = std::accumulate(stats.global_vertex_counts.begin(), stats.global_vertex_counts.end(), 0);
    stats.total_edges    = std::accumulate(stats.global_edge_counts.begin(), stats.global_edge_counts.end(), 0);
    stats.expected_vertices = static_cast<double>(stats.total_vertices) / num_partitions;
    stats.expected_edges    = (stats.total_edges > 0) ? static_cast<double>(stats.total_edges) / num_partitions : 1.0;
    return stats;
}

/**
 * @brief MPI Delta 통신을 위한 최적화된 Allgather 함수 (구조체 안전 타입)
 */
static std::vector<Delta> allgatherDeltas(const std::vector<Delta> &local_deltas, int mpi_size) {
    int send_count = (int)local_deltas.size();
    std::vector<int> recv_counts(mpi_size);
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(mpi_size);
    displs[0] = 0;
    for (int i = 1; i < mpi_size; i++) displs[i] = displs[i - 1] + recv_counts[i - 1];
    int total_recv = displs[mpi_size - 1] + recv_counts[mpi_size - 1];
    std::vector<Delta> recv_deltas(total_recv);

    // Delta용 MPI 타입 (구조체 패딩 안전)
    MPI_Datatype MPI_DELTA;
    int block_lengths[2] = {1, 1};
    MPI_Aint disps[2];
    MPI_Aint base, addr_gid, addr_label;
    Delta tmp;
    MPI_Get_address(&tmp, &base);
    MPI_Get_address(&tmp.gid, &addr_gid);
    MPI_Get_address(&tmp.new_label, &addr_label);
    disps[0] = addr_gid   - base;
    disps[1] = addr_label - base;
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Type_create_struct(2, block_lengths, disps, types, &MPI_DELTA);
    MPI_Type_commit(&MPI_DELTA);

    MPI_Allgatherv(local_deltas.data(), send_count, MPI_DELTA,
                   recv_deltas.data(), recv_counts.data(), displs.data(),
                   MPI_DELTA, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_DELTA);
    return recv_deltas;
}

// ==================== DMOLP Penalty 계산 ====================

std::vector<std::vector<double>> calculatePenalties(
    const PartitionStats &stats,
    int num_partitions,
    int iter,
    int mpi_rank = 0)
{
    std::vector<std::vector<double>> result(2, std::vector<double>(num_partitions, 0.0));

    if (mpi_rank == 0) {
        std::vector<double> RV(num_partitions), RE(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            RV[i] = (stats.expected_vertices > 0) ? static_cast<double>(stats.global_vertex_counts[i]) / stats.expected_vertices : 1.0;
            RE[i] = (stats.expected_edges > 0) ? static_cast<double>(stats.global_edge_counts[i]) / stats.expected_edges : 1.0;
        }

        printf("\n=== Label Statistics (Master-Worker 방식) ===\n");
        printf("Total: %d vertices, %d edges\n", stats.total_vertices, stats.total_edges);
        printf("Expected per label: %.1f vertices, %.1f edges\n", stats.expected_vertices, stats.expected_edges);
        for (int i = 0; i < num_partitions; i++) {
            printf("Label %d: %d vertices (%.3f), %d edges (%.3f)\n",
                   i, stats.global_vertex_counts[i], RV[i],
                   stats.global_edge_counts[i], RE[i]);
        }
        printf("========================\n\n");

        double rv_mean = 0.0, re_mean = 0.0;
        for (int i = 0; i < num_partitions; i++) { rv_mean += RV[i]; re_mean += RE[i]; }
        rv_mean /= num_partitions; re_mean /= num_partitions;

        double rv_var = 0.0, re_var = 0.0;
        for (int i = 0; i < num_partitions; i++) {
            rv_var += (RV[i] - rv_mean) * (RV[i] - rv_mean);
            re_var += (RE[i] - re_mean) * (RE[i] - re_mean);
        }
        rv_var /= num_partitions; re_var /= num_partitions;

        double total_var = rv_var + re_var;
        double imb_rv = (total_var > 0) ? rv_var / total_var : 0.0;
        double imb_re = (total_var > 0) ? re_var / total_var : 0.0;

        for (int i = 0; i < num_partitions; i++) {
            double G_RV = (1.0 - RV[i]) * (1.0 + log(iter));
            double G_RE = (1.0 - RE[i]) * (1.0 + log(iter));
            result[0][i] = imb_rv * G_RV + imb_re * G_RE;
            result[1][i] = RE[i];
        }
    }

    MPI_Bcast(result[0].data(), num_partitions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(result[1].data(), num_partitions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return result;
}

/**
 * @brief Boundary Local ID 추출 함수 (현재 코드에선 미사용 가능)
 */
static std::vector<int> extractBoundaryLocalIDs(const Graph &graph, const GhostNodes &ghost_nodes) {
    std::vector<int> boundary_nodes;
    #pragma omp parallel
    {
        std::vector<int> thread_boundary_nodes;
        #pragma omp for nowait
        for (int u = 0; u < graph.num_vertices; u++) {
            int u_label = graph.vertex_labels[u];
            bool is_boundary = false;
            for (int edge_idx = graph.row_ptr[u]; edge_idx < graph.row_ptr[u + 1]; edge_idx++) {
                int v = graph.col_indices[edge_idx];
                int v_label = getNodeLabel(v, graph, graph.vertex_labels, ghost_nodes);
                if (v_label != -1 && u_label != v_label) { is_boundary = true; break; }
            }
            if (is_boundary) thread_boundary_nodes.push_back(u);
        }
        #pragma omp critical
        {
            boundary_nodes.insert(boundary_nodes.end(), thread_boundary_nodes.begin(), thread_boundary_nodes.end());
        }
    }
    return boundary_nodes;
}

/**
 * @brief Edge-cut 계산 함수
 */
static int computeEdgeCut(const Graph &g, const std::vector<int> &labels, const GhostNodes &ghost_nodes, int mpi_rank) {
    int local_cut = 0;
    int total_edges = 0;
    int invalid_label_pairs = 0;

    #pragma omp parallel reduction(+:local_cut,total_edges,invalid_label_pairs)
    {
        #pragma omp for nowait
        for (int u = 0; u < g.num_vertices; u++) {
            for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; e++) {
                int v = g.col_indices[e];
                total_edges++;
                int u_label = labels[u];
                int v_label = getNodeLabel(v, g, labels, ghost_nodes);
                if (u_label == -1 || v_label == -1) { invalid_label_pairs++; continue; }
                if (u_label != v_label) local_cut++;
            }
        }
    }

    int global_cut = 0, global_total_edges = 0, global_invalid = 0;
    MPI_Allreduce(&local_cut, &global_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_edges, &global_total_edges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&invalid_label_pairs, &global_invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_invalid != 0) {
        if (mpi_rank == 0) {
            printf("[EdgeCut] Detected %d invalid label pairs across ranks. Aborting.\n", global_invalid);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    if ( (global_cut & 1) != 0 ) {
        printf("[Rank %d] Local edge-cut: %d, Global edge-cut: %d\n", mpi_rank, local_cut, global_cut);
        printf("Error: Global edge-cut is odd (%d). This indicates a ghost-label sync inconsistency.\n", global_cut);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return (global_cut >> 1);
}

// ==================== Phase2 메인 ====================
PartitioningMetrics run_phase2(
    int mpi_rank, int mpi_size,
    int num_partitions,
    Graph &local_graph,
    GhostNodes &ghost_nodes,
    int gpu_id)
{
    const int max_iter = 500;
    const double epsilon = 0.03;
    const int k_limit = 10;

    auto t_phase2_start = std::chrono::high_resolution_clock::now();

    std::cout << "[Rank " << mpi_rank << "/" << mpi_size << "] Phase2 시작 (GPU " << gpu_id << ")" << std::endl;
    std::cout.flush();

    // 핀/버퍼
    std::vector<double> penalty_pinned;
    std::vector<double> RE_pinned;
    std::vector<int> boundary_nodes_pinned;
    penalty_pinned.reserve(num_partitions);

    // 초기 상태
    printf("[Rank %d] Phase2 초기 edge-cut 계산 시작...\n", mpi_rank);
    int prev_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels, ghost_nodes, mpi_rank);
    int convergence_count = 0;

    PartitionStats current_stats = computePartitionStats(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions);

    penalty_pinned.resize(num_partitions);
    RE_pinned.resize(num_partitions);
    std::vector<Delta> delta_changes; delta_changes.reserve(1000);

    std::vector<int> current_boundary_nodes;
    current_boundary_nodes.clear();
    for (int i = 0; i < local_graph.num_vertices; i++) current_boundary_nodes.push_back(i);

    PartitioningMetrics m2;

    for (int iter = 0; iter < max_iter; iter++) {
        // Step 1: Penalty 계산
        auto penalties = calculatePenalties(current_stats, num_partitions, iter + 1, mpi_rank);
        penalty_pinned = penalties[0];
        RE_pinned      = penalties[1];

        // Step 2: 바운더리 확장
        current_boundary_nodes = expandBoundaryNodes(
            local_graph.row_ptr, local_graph.col_indices,
            current_boundary_nodes, local_graph.vertex_labels,
            penalty_pinned, RE_pinned,
            local_graph.num_vertices + ghost_nodes.ghost_labels.size(),
            iter + 1);

        if (current_boundary_nodes.empty()) {
            if (mpi_rank == 0) std::cout << "경계 노드 없음, 수렴 완료\n";
            break;
        }

        // Step 3: GPU 라벨 전파
        GPULabelUpdateResult gpu_result;
        try {
            size_t free_memory, total_memory; cudaMemGetInfo(&free_memory, &total_memory);
            size_t max_gpu_memory = static_cast<size_t>(free_memory * 0.85);
            printf("[Rank %d] GPU 메모리: 전체 %.1fGB, 사용가능 %.1fGB, 사용예정 %.1fGB\n",
                   mpi_rank, total_memory / (1024.0*1024.0*1024.0),
                   free_memory / (1024.0*1024.0*1024.0),
                   max_gpu_memory / (1024.0*1024.0*1024.0));

            auto gpu_start = std::chrono::high_resolution_clock::now();
            gpu_result = runBoundaryLPOnGPU_Streaming(
                local_graph.row_ptr,
                local_graph.col_indices,
                current_boundary_nodes,
                local_graph.vertex_labels,
                ghost_nodes.ghost_labels,
                local_graph.global_ids,
                penalty_pinned,
                local_graph.num_vertices,
                num_partitions,
                max_gpu_memory / (1024 * 1024)
            );
            auto gpu_end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
            printf("[Rank %d] GPU 처리 완료: %.2fms, %d 변경사항\n", mpi_rank, dur.count()/1000.0, gpu_result.change_count);
        } catch (const std::exception& e) {
            printf("[Rank %d] GPU 처리 실패 (CPU fallback 미구현): %s\n", mpi_rank, e.what());
            gpu_result.change_count = 0;
        }

        // GPU 작업 동기화
        cudaDeviceSynchronize();

        // Step 4: 로컬 delta 구축 (GPU 결과 → delta_changes)
        delta_changes.clear();
        for (int i = 0; i < gpu_result.change_count; i++) {
            int local_node_id = gpu_result.updated_nodes[i];
            int new_label     = gpu_result.updated_labels[i];
            if (0 <= local_node_id && local_node_id < local_graph.num_vertices) {
                if (local_node_id < (int)local_graph.global_ids.size()) {
                    Delta d; d.gid = local_graph.global_ids[local_node_id]; d.new_label = new_label;
                    delta_changes.push_back(d);
                }
                local_graph.vertex_labels[local_node_id] = new_label;
            }
        }
        std::cout << "[Rank " << mpi_rank << "] GPU 라벨 변경: " << gpu_result.change_count
                  << " (로컬 delta: " << delta_changes.size() << ")" << std::endl;

        // 현재 이터레이션 변경분을 수집 (동기 Allgather)
        auto recv_deltas = allgatherDeltas(delta_changes, mpi_size);
        printf("[Rank %d] Iter %d: 로컬 변경 %zu개, 수신 변경 %zu개\n",
               mpi_rank, iter, delta_changes.size(), recv_deltas.size());

        // 수신 델타 적용 (ghost 라벨 업데이트)
        for (const auto &delta : recv_deltas) {
            auto it_ghost = ghost_nodes.global_to_local.find(delta.gid);
            if (it_ghost != ghost_nodes.global_to_local.end()) {
                int ghost_idx = it_ghost->second;
                if (0 <= ghost_idx && ghost_idx < (int)ghost_nodes.ghost_labels.size()) {
                    ghost_nodes.ghost_labels[ghost_idx] = delta.new_label;
                    int ghost_lid = local_graph.num_vertices + ghost_idx;
                    if (ghost_lid < (int)local_graph.vertex_labels.size()) {
                        local_graph.vertex_labels[ghost_lid] = delta.new_label;
                    }
                }
            }
        }

        // 모든 랭크가 동일 스냅샷을 보게 보장
        MPI_Barrier(MPI_COMM_WORLD);

        // 전역 변경 수 계산 및 조기 종료 판정
        int total_changes = (int)delta_changes.size() + (int)recv_deltas.size();
        int global_total_changes = 0;
        MPI_Allreduce(&total_changes, &global_total_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_total_changes == 0) {
            if (mpi_rank == 0) std::cout << "Iter " << (iter + 1) << ": 전체 시스템 수렴 완료 (변경사항 없음)\n";
            break;
        }

        // Step 6: 통계/수렴 판정 (Barrier 이후)
        current_stats = computePartitionStats(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions);

        printf("[Rank %d] Iter %d: Edge-cut 계산 중...\n", mpi_rank, iter + 1);
        int curr_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels, ghost_nodes, mpi_rank);
        double delta = (prev_edge_cut > 0) ? std::abs((double)(curr_edge_cut - prev_edge_cut) / prev_edge_cut) : 1.0;

        if (mpi_rank == 0) {
            std::cout << "Iter " << (iter + 1) << ": Edge-cut " << curr_edge_cut
                      << " (변화율: " << std::fixed << std::setprecision(3) << delta * 100 << "%)";
            if (delta < epsilon) std::cout << " [수렴 진행: " << (convergence_count + 1) << "/" << k_limit << "]";
            std::cout << "\n";
        }

        if (delta < epsilon) convergence_count++; else convergence_count = 0;
        prev_edge_cut = curr_edge_cut;

        if (convergence_count >= k_limit) {
            if (mpi_rank == 0) {
                std::cout << "수렴 완료! (연속 " << k_limit << "회 변화율 < "
                          << std::fixed << std::setprecision(1) << (epsilon * 100) << "%)\n";
            }
            break;
        }
    }

    auto t_phase2_end = std::chrono::high_resolution_clock::now();
    long exec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_phase2_end - t_phase2_start).count();

    std::cout << "[Rank " << mpi_rank << "] Phase2 완료 - GPU " << gpu_id
              << " 총 실행시간: " << exec_ms << "ms" << std::endl;

    PartitionStats final_stats = computePartitionStats(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions);

    double max_vertex_ratio = 0.0, max_edge_ratio = 0.0;
    for (int i = 0; i < num_partitions; i++) {
        double rv = (final_stats.expected_vertices > 0) ? static_cast<double>(final_stats.global_vertex_counts[i]) / final_stats.expected_vertices : 1.0;
        double re = (final_stats.expected_edges > 0) ? static_cast<double>(final_stats.global_edge_counts[i]) / final_stats.expected_edges : 1.0;
        max_vertex_ratio = std::max(max_vertex_ratio, rv);
        max_edge_ratio   = std::max(max_edge_ratio, re);
    }

    m2.edge_cut = prev_edge_cut;
    m2.vertex_balance = max_vertex_ratio;
    m2.edge_balance = max_edge_ratio;
    m2.loading_time_ms = exec_ms;
    m2.distribution_time_ms = 0;
    m2.num_partitions = num_partitions;
    m2.total_vertices = final_stats.total_vertices;
    m2.total_edges = final_stats.total_edges;

    MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();
    std::cout << "[Rank " << mpi_rank << "] GPU " << gpu_id << " 작업 완료" << std::endl;
    std::cout.flush();

    return m2;
}
