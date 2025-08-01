#include "phase2.h"
#include "gpu_lp_boundary.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "graph_types.h"
#include <iomanip>

void calculatePartitionRatios(
    const Graph &g,
    const std::vector<int> &labels,
    int num_partitions,
    std::vector<PartitionInfo> &PI)
{
    std::vector<int> local_vertex_counts(num_partitions, 0);
    std::vector<int> local_edge_counts(num_partitions, 0);

    // 정점/간선 카운트
    for (int u = 0; u < g.num_vertices; u++) {
        int label = labels[u];
        local_vertex_counts[label]++;
        for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; e++) {
            int v = g.col_indices[e];
            if (labels[v] == label)
                local_edge_counts[label]++;
        }
    }

    std::vector<int> global_vertex_counts(num_partitions, 0);
    std::vector<int> global_edge_counts(num_partitions, 0);

    MPI_Allreduce(local_vertex_counts.data(), global_vertex_counts.data(),
                  num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_edge_counts.data(), global_edge_counts.data(),
                  num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int total_vertices = std::accumulate(global_vertex_counts.begin(), global_vertex_counts.end(), 0);
    int total_edges = std::accumulate(global_edge_counts.begin(), global_edge_counts.end(), 0);

    double expected_vertices = static_cast<double>(total_vertices) / num_partitions;
    double expected_edges = (total_edges > 0) ? static_cast<double>(total_edges) / num_partitions : 1.0;

    for (int i = 0; i < num_partitions; i++) {
        PI[i].RV = static_cast<double>(global_vertex_counts[i]) / expected_vertices;
        PI[i].RE = static_cast<double>(global_edge_counts[i]) / expected_edges;
    }
}

static void calculatePenalty(std::vector<PartitionInfo> &PI, int num_partitions)
{
    double rv_mean = 0.0, re_mean = 0.0;
    for (auto &p : PI) {
        rv_mean += p.RV;
        re_mean += p.RE;
    }
    rv_mean /= num_partitions;
    re_mean /= num_partitions;

    double rv_var = 0.0, re_var = 0.0;
    for (auto &p : PI) {
        rv_var += (p.RV - rv_mean) * (p.RV - rv_mean);
        re_var += (p.RE - re_mean) * (p.RE - re_mean);
    }
    rv_var /= num_partitions;
    re_var /= num_partitions;

    double total_var = rv_var + re_var;
    double imb_rv = (total_var > 0) ? rv_var / total_var : 0.0;
    double imb_re = (total_var > 0) ? re_var / total_var : 0.0;

    for (auto &p : PI) {
        p.imb_RV = imb_rv;
        p.imb_RE = imb_re;
        p.G_RV = (1.0 - p.RV) / num_partitions;
        p.G_RE = (1.0 - p.RE) / num_partitions;
        p.P_L = imb_rv * p.G_RV + imb_re * p.G_RE;
    }
}

static std::vector<int> extractBoundaryGlobalIDs(
    const Graph &g,
    const std::vector<int> &labels,
    const std::vector<int> &global_ids)
{
    std::vector<int> boundary;
    boundary.reserve(g.num_vertices / 10); // 대략적 초기 크기
    for (int u = 0; u < g.num_vertices; u++) {
        int label_u = labels[u];
        bool is_boundary = false;
        for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; e++) {
            int v = g.col_indices[e];
            if (labels[v] != label_u) {
                is_boundary = true;
                break;
            }
        }
        if (is_boundary) boundary.push_back(global_ids[u]);
    }
    return boundary;
}

static int computeEdgeCut(const Graph &g, const std::vector<int> &labels)
{
    int local_cut = 0;
    for (int u = 0; u < g.num_vertices; u++) {
        for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; e++) {
            int v = g.col_indices[e];
            if (labels[u] != labels[v])
                local_cut++;
        }
    }
    int global_cut = 0;
    MPI_Allreduce(&local_cut, &global_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_cut / 2;
}

void run_phase2(
    int mpi_rank, int mpi_size,
    int num_partitions,
    Graph &local_graph,
    std::vector<int> &vertex_labels,
    const std::vector<int> &global_ids,
    const std::unordered_map<int, int> &global_to_local,
    const Phase1Metrics &metrics)
{
    const int max_iter = 500;
    const double epsilon = 0.03;
    const int k_limit = 10;

    std::vector<int> labels_new = vertex_labels;
    std::vector<PartitionInfo> PI(num_partitions);
    std::vector<double> penalty(num_partitions, 0.0);

    int prev_edge_cut = computeEdgeCut(local_graph, vertex_labels);
    int stable_count = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        if (mpi_rank == 0)
            std::cout << "=== Iter " << iter + 1 << " ===\n";

        // Step1: RV, RE 계산
        calculatePartitionRatios(local_graph, vertex_labels, num_partitions, PI);
        if (mpi_rank == 0) {
            std::cout << "RV, RE 계산 완료\n";
            for (int i = 0; i < num_partitions; i++) {
                std::cout << "Partition " << i << ": RV = " << PI[i].RV
                          << ", RE = " << PI[i].RE
                          << ", P_L = " << PI[i].P_L << "\n";
            }
        }

        // Step2: Penalty 계산
        calculatePenalty(PI, num_partitions);
        for (int i = 0; i < num_partitions; i++)
            penalty[i] = PI[i].P_L;
        if (mpi_rank == 0) {
            std::cout << "Penalty 계산 완료\n";
            for (int i = 0; i < num_partitions; i++) {
                std::cout << "Partition " << i << ": P_L = " << penalty[i] << "\n";
            }
        // Step3: BV 추출
        auto boundary_nodes_gid = extractBoundaryGlobalIDs(local_graph, vertex_labels, global_ids);
        if (mpi_rank == 0) {
            std::cout << "Boundary 노드 추출 완료: " << boundary_nodes_gid.size() << "개\n";
        }
        // Step3b: 로컬 인덱스 변환
        std::vector<int> boundary_nodes_local;
        boundary_nodes_local.reserve(boundary_nodes_gid.size());
        for (int gid : boundary_nodes_gid) {
            auto it = global_to_local.find(gid);
            if (it != global_to_local.end())
                boundary_nodes_local.push_back(it->second);
        }
        if (mpi_rank == 0) {
            std::cout << "로컬 경계 노드 변환 완료: " << boundary_nodes_local.size() << "개\n";
        }
        // Step4: GPU 커널 실행 (Warp Reduction 버전)
        runBoundaryLPOnGPU_Warp(local_graph.row_ptr, local_graph.col_indices,
                                vertex_labels, labels_new, penalty,
                                boundary_nodes_local, num_partitions);
        if (mpi_rank == 0) {
            std::cout << "GPU LP 실행 완료\n";
            // std::cout << "변경된 라벨: ";
            // for (int i = 0; i < labels_new.size(); i++) {
            //     std::cout << labels_new[i] << " ";
            // }
            std::cout << "\n";
        }
        
        // Step5: 경계노드 교환
        int send_count = boundary_nodes_gid.size();
        std::vector<int> recv_counts(mpi_size);
        MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        int total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
        std::vector<int> recv_buf(total_recv);
        std::vector<int> displs(mpi_size);
        displs[0] = 0;
        for (int i = 1; i < mpi_size; i++)
            displs[i] = displs[i - 1] + recv_counts[i - 1];

        MPI_Allgatherv(boundary_nodes_gid.data(), send_count, MPI_INT,
                       recv_buf.data(), recv_counts.data(), displs.data(),
                       MPI_INT, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            std::cout << "경계 노드 교환 완료: 총 " << total_recv << "개\n";
        }   
        // Step5b: 변경된 라벨 적용
        for (int gid : recv_buf) {
            auto it = global_to_local.find(gid);
            if (it != global_to_local.end()) {
                int lid = it->second;
                vertex_labels[lid] = labels_new[lid];
            }
        }
        if (mpi_rank == 0) {
            std::cout << "변경된 라벨 적용 완료\n";
        }
        // Step6: Edge-cut 변화율 검사
        int curr_edge_cut = computeEdgeCut(local_graph, vertex_labels);
        double delta = (prev_edge_cut > 0)
                           ? std::abs((double)(curr_edge_cut - prev_edge_cut) / prev_edge_cut)
                           : 1.0;
        if (delta < epsilon)
            stable_count++;
        else
            stable_count = 0;

        prev_edge_cut = curr_edge_cut;

        if (stable_count >= k_limit) {
            if (mpi_rank == 0) std::cout << "수렴!\n";
            break;
        }
        if (mpi_rank == 0) {
            std::cout << "현재 Edge-cut: " << curr_edge_cut << ", 변화율: " << delta << "\n";
        }
    }
        if (mpi_rank == 0) {
            std::cout << "Phase 2 완료 - 최종 Edge-cut: " << prev_edge_cut << "\n";
            std::cout << "각 파티션의 라벨 분포:\n";
            for (int i = 0; i < num_partitions; i++) {
                int count = std::count(vertex_labels.begin(), vertex_labels.end(), i);
                std::cout << "Partition " << i << ": " << count << "개\n";
            }
        }
    }
    // Phase 1 메트릭 출력
    if (mpi_rank == 0) {
        std::cout << "Phase 1 메트릭:\n";
        std::cout << "초기 Edge-cut: " << metrics.initial_edge_cut << "\n";
        std::cout << "초기 Vertex Balance: " << metrics.initial_vertex_balance << "\n";
        std::cout << "초기 Edge Balance: " << metrics.initial_edge_balance << "\n";
        std::cout << "총 정점 수: " << metrics.total_vertices << "\n";
        std::cout << "총 간선 수: " << metrics.total_edges << "\n";
        std::cout << "파티션별 정점 수: ";
        for (int count : metrics.partition_vertex_counts) {
            std::cout << count << " ";
        }
        std::cout << "\n";
        std::cout << "파티션별 간선 수: ";
        for (int count : metrics.partition_edge_counts) {
            std::cout << count << " ";
        }
        std::cout << "\n";
        std::cout << "로딩 시간 (ms): " << metrics.loading_time_ms << "\n";
        std::cout << "분배 시간 (ms): " << metrics.distribution_time_ms << "\n";
        std::cout << "총 실행 시간 (ms): " << (metrics.loading_time_ms + metrics.distribution_time_ms) << "\n";

        std::cout << "Phase 2 실행 시간 (ms): " << (metrics.loading_time_ms + metrics.distribution_time_ms) << "\n";
        std::cout << "총 소요 시간 (ms): "
                  << (metrics.loading_time_ms + metrics.distribution_time_ms +
                      (metrics.loading_time_ms + metrics.distribution_time_ms)) << "\n";
        std::cout << "Phase 2 완료 - 최종 Edge-cut: " << prev_edge_cut << "\n";
        std::cout << "각 파티션의 라벨 분포:\n";
        for (int i = 0; i < num_partitions; i++) {
            int count = std::count(vertex_labels.begin(), vertex_labels.end(), i);
            std::cout << "Partition " << i << ": " << count << "개\n";
        }

        std::cout << "\n=== Phase 1 vs Phase 2 (7단계 알고리즘) 비교 ===\n";
        double edge_cut_improvement = 0.0;
        if (metrics.initial_edge_cut > 0) {
            edge_cut_improvement = (static_cast<double>(metrics.initial_edge_cut - prev_edge_cut) / metrics.initial_edge_cut) * 100.0;
        }
        double vertex_balance_improvement = 0.0;
        if (metrics.initial_vertex_balance > 0) {
            vertex_balance_improvement = ((metrics.initial_vertex_balance - metrics.initial_edge_balance) / metrics.initial_vertex_balance) * 100.0;
        }
        double edge_balance_improvement = 0.0;
        if (metrics.initial_edge_balance > 0) {
            edge_balance_improvement = ((metrics.initial_edge_balance - metrics.initial_edge_balance) / metrics.initial_edge_balance) * 100.0;
        }
        std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "│                    메트릭 비교 결과                         │\n";
        std::cout << "├─────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Edge-cut:                                                   │\n";
        std::cout << "│   Phase 1 (초기): " << std::setw(10) << metrics.initial_edge_cut << "                              │\n";
        std::cout << "│   Phase 2 (최종): " << std::setw(10) << prev_edge_cut << "                              │\n";
        std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_cut_improvement << "%                             │\n";
        std::cout << "│                                                             │\n";
        std::cout << "│ Vertex Balance:                                             │\n";
        std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << metrics.initial_vertex_balance << "                             │\n";
        std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << metrics.initial_edge_balance << "                             │\n";
        std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << vertex_balance_improvement << "%                             │\n";
        std::cout << "│                                                             │\n";
        std::cout << "│ Edge Balance:                                               │\n";
        std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << metrics.initial_edge_balance << "                             │\n";
        std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << metrics.initial_edge_balance << "                             │\n";
        std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_balance_improvement << "%                             │\n";
        std::cout << "│                                                             │\n";
        std::cout << "│ 실행 시간:                                                  │\n";
        std::cout << "│   Phase 1 (로딩): " << std::setw(6) << metrics.loading_time_ms << " ms                          │\n";
        std::cout << "│   Phase 1 (분산): " << std::setw(6) << metrics.distribution_time_ms << " ms                          │\n";
        std::cout << "│   Phase 2 (7단계): " << std::setw(5) << (metrics.loading_time_ms + metrics.distribution_time_ms) << " ms                          │\n";
        std::cout << "│   총 소요시간:     " << std::setw(5) << (metrics.loading_time_ms + metrics.distribution_time_ms + (metrics.loading_time_ms + metrics.distribution_time_ms)) << " ms                          │\n";
        std::cout << "└─────────────────────────────────────────────────────────────┘\n";
        std::cout << "\n=== 알고리즘 성능 요약 ===\n";
        if (edge_cut_improvement > 0) {
            std::cout << "✓ Edge-cut " << std::fixed << std::setprecision(1) << edge_cut_improvement << "% 개선 ("
                      << metrics.initial_edge_cut << " → " << prev_edge_cut << ")\n";
        } else {
            std::cout << "⚠ Edge-cut " << std::fixed << std::setprecision(1) << -edge_cut_improvement << "% 악화 ("
                      << metrics.initial_edge_cut << " → " << prev_edge_cut << ")\n";
        }
        if (vertex_balance_improvement > 0) {
            std::cout << "✓ Vertex Balance " << std::fixed << std::setprecision(1) << vertex_balance_improvement << "% 개선 ("
                      << metrics.initial_vertex_balance << " → " << metrics.initial_edge_balance << ")\n";
        } else {
            std::cout << "⚠ Vertex Balance " << std::fixed << std::setprecision(1) << -vertex_balance_improvement << "% 악화 ("
                      << metrics.initial_vertex_balance << " → " << metrics.initial_edge_balance << ")\n";
        }
        if (edge_balance_improvement > 0) {
            std::cout << "✓ Edge Balance " << std::fixed << std::setprecision(1) << edge_balance_improvement << "% 개선 ("
                      << metrics.initial_edge_balance << " → " << metrics.initial_edge_balance << ")\n";
        } else {
            std::cout << "⚠ Edge Balance " << std::fixed << std::setprecision(1) << -edge_balance_improvement << "% 악화 ("
                      << metrics.initial_edge_balance << " → " << metrics.initial_edge_balance << ")\n";
        }
        std::cout << "총 소요시간: " << (metrics.loading_time_ms + metrics.distribution_time_ms + (metrics.loading_time_ms + metrics.distribution_time_ms)) << " ms\n";
    }


}