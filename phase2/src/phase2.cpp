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
#include "report_utils.h"
#include <iomanip>
#include <chrono>

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

PartitioningMetrics run_phase2(
    int mpi_rank, int mpi_size,
    int num_partitions,
    Graph &local_graph,
    std::vector<int> &vertex_labels,
    const std::vector<int> &global_ids,
    const std::unordered_map<int, int> &global_to_local)
{
    const int max_iter = 500;
    const double epsilon = 0.03;
    const int k_limit = 10;

    auto t_phase2_start = std::chrono::high_resolution_clock::now();

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

    auto t_phase2_end = std::chrono::high_resolution_clock::now();
    long exec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_phase2_end - t_phase2_start).count();
    // Vertex/Edge Balance 계산 (최대값/평균값)
    double max_vertex_ratio = 0.0, max_edge_ratio = 0.0;
    for (const auto &p : PI) {
        max_vertex_ratio = std::max(max_vertex_ratio, p.RV);
        max_edge_ratio = std::max(max_edge_ratio, p.RE);
    }
    double avg_vertex_ratio = std::accumulate(PI.begin(), PI.end(), 0.0, [](double acc, const PartitionInfo& p){ return acc + p.RV; }) / num_partitions;
    double avg_edge_ratio = std::accumulate(PI.begin(), PI.end(), 0.0, [](double acc, const PartitionInfo& p){ return acc + p.RE; }) / num_partitions;
    // Vertex Balance: 최대값 / 평균값 (1에 가까울수록 균형)
    double final_vertex_balance = max_vertex_ratio / avg_vertex_ratio;
    // Edge Balance: 최대값 / 평균값 (1에 가까울수록 균형)
    double final_edge_balance = max_edge_ratio / avg_edge_ratio;
    

    PartitioningMetrics m2;
    m2.edge_cut = prev_edge_cut;
    m2.vertex_balance = final_vertex_balance;
    m2.edge_balance = final_edge_balance;
    m2.loading_time_ms = exec_ms; // Phase2 실행시간
    m2.distribution_time_ms = 0;
    m2.num_partitions = num_partitions;
    return m2;
}