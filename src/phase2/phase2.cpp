#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <chrono>

#include "graph_types.h"
#include "phase2/phase2.h"
#include "phase2/gpu_lp_boundary.h"

// === Partition 비율 계산 ===
void calculatePartitionRatios(
    const Graph &g,
    const std::vector<int> &labels,
    int num_partitions,
    std::vector<PartitionInfo> &PI)
{
    std::vector<int> local_vertex_counts(num_partitions, 0);
    std::vector<int> local_edge_counts(num_partitions, 0);

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

// === Penalty 계산 ===
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

// === 경계 노드 global id 추출 ===
static std::vector<int> extractBoundaryGlobalIDs(
    const Graph &g,
    const std::vector<int> &labels)
{
    std::vector<int> boundary;
    boundary.reserve(g.num_vertices / 10);
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
        if (is_boundary) boundary.push_back(g.global_ids[u]); // local index → global id
    }
    return boundary;
}

// === Edge-cut 계산 ===
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

// === Phase2 실행 ===
PartitioningMetrics run_phase2(
    int mpi_rank, int mpi_size,
    int num_partitions,
    Graph &local_graph,
    GhostNodes &ghost_nodes)
{
    const int max_iter = 500;
    const double epsilon = 0.03;
    const int k_limit = 10;

    auto t_phase2_start = std::chrono::high_resolution_clock::now();

    std::vector<int> labels_new = local_graph.vertex_labels; // 현재 라벨 복사
    std::vector<PartitionInfo> PI(num_partitions);
    std::vector<double> penalty(num_partitions, 0.0);

    int prev_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels);
    int stable_count = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        // labels_new를 현재 라벨로 동기화
        labels_new = local_graph.vertex_labels;
        // Step1: RV, RE 계산
        calculatePartitionRatios(local_graph, local_graph.vertex_labels, num_partitions, PI);

        // Step2: Penalty 계산
        calculatePenalty(PI, num_partitions);
        for (int i = 0; i < num_partitions; i++)
            penalty[i] = PI[i].P_L;
            
        // Penalty 디버깅
        if (mpi_rank == 0 && iter < 3) {
            std::cout << "  [DEBUG] Penalty 값들: ";
            for (int i = 0; i < num_partitions; i++) {
                std::cout << "P[" << i << "]=" << std::fixed << std::setprecision(4) << penalty[i] << " ";
            }
            std::cout << "\n";
        }

        // Step3: Boundary 노드 추출(global id)
        auto boundary_nodes_gid = extractBoundaryGlobalIDs(local_graph, local_graph.vertex_labels);

        // Step3b: global → local index 변환
        std::vector<int> boundary_nodes_local;
        boundary_nodes_local.reserve(boundary_nodes_gid.size());
        for (int gid : boundary_nodes_gid) {
            // 1) 내 소유 노드? (Graph.global_ids에서 찾기)
            auto it_owned = std::find(local_graph.global_ids.begin(), local_graph.global_ids.end(), gid);
            if (it_owned != local_graph.global_ids.end()) {
                int lid = (int)(it_owned - local_graph.global_ids.begin());
                boundary_nodes_local.push_back(lid);
            } else {
                // 2) ghost node? (GhostNodes.global_to_local 매핑)
                auto it_ghost = ghost_nodes.global_to_local.find(gid);
                if (it_ghost != ghost_nodes.global_to_local.end())
                    boundary_nodes_local.push_back(it_ghost->second);
            }
        }

        // Step4: GPU 커널 실행
        if (mpi_rank == 0 && iter < 3) {
            std::cout << "  [DEBUG] GPU 커널 실행 전 boundary_nodes_local 수: " << boundary_nodes_local.size() << "\n";
            if (boundary_nodes_local.size() > 0) {
                std::cout << "  [DEBUG] 첫 5개 경계 노드 라벨 (실행 전): ";
                for (int i = 0; i < std::min(5, (int)boundary_nodes_local.size()); i++) {
                    int lid = boundary_nodes_local[i];
                    std::cout << "(" << lid << ":" << local_graph.vertex_labels[lid] << ") ";
                }
                std::cout << "\n";
            }
        }
        
        runBoundaryLPOnGPU_Warp(local_graph.row_ptr,
                                local_graph.col_indices,
                                local_graph.vertex_labels, // old labels
                                labels_new,                // new labels
                                penalty,
                                boundary_nodes_local,
                                num_partitions);

        if (mpi_rank == 0 && iter < 3) {
            if (boundary_nodes_local.size() > 0) {
                std::cout << "  [DEBUG] 첫 5개 경계 노드 라벨 (실행 후): ";
                for (int i = 0; i < std::min(5, (int)boundary_nodes_local.size()); i++) {
                    int lid = boundary_nodes_local[i];
                    std::cout << "(" << lid << ":" << labels_new[lid] << ") ";
                }
                std::cout << "\n";
            }
        }

        // Step4b: GPU 커널 결과를 local_graph.vertex_labels에 적용
        int label_changes = 0;
        for (int lid : boundary_nodes_local) {
            if (local_graph.vertex_labels[lid] != labels_new[lid]) {
                label_changes++;
            }
            local_graph.vertex_labels[lid] = labels_new[lid];
        }
        
        if (mpi_rank == 0 && iter < 3) {
            std::cout << "  [DEBUG] 라벨 변경된 노드 수: " << label_changes << "/" << boundary_nodes_local.size() << "\n";
        }

        // Step5: 변경된 라벨 정보만 전송 (Delta 구조체 사용)
        std::vector<Delta> delta_changes;
        for (int lid : boundary_nodes_local) {
            Delta delta;
            delta.gid = local_graph.global_ids[lid];
            delta.new_label = labels_new[lid];
            delta_changes.push_back(delta);
        }

        // Allgather 준비
        int send_count = delta_changes.size();
        std::vector<int> recv_counts(mpi_size);
        MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> displs(mpi_size);
        displs[0] = 0;
        for (int i = 1; i < mpi_size; i++)
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        int total_recv = displs[mpi_size - 1] + recv_counts[mpi_size - 1];

        std::vector<Delta> recv_deltas(total_recv);

        // Delta용 MPI 타입 정의
        MPI_Datatype MPI_DELTA;
        MPI_Type_contiguous(2, MPI_INT, &MPI_DELTA); // gid + new_label
        MPI_Type_commit(&MPI_DELTA);

        MPI_Allgatherv(delta_changes.data(), send_count, MPI_DELTA,
                       recv_deltas.data(), recv_counts.data(), displs.data(),
                       MPI_DELTA, MPI_COMM_WORLD);

        MPI_Type_free(&MPI_DELTA);

        // Step5b: 수신된 라벨 변경사항 적용
        for (const auto &delta : recv_deltas) {
            // ghost 노드인지 확인
            auto it_ghost = ghost_nodes.global_to_local.find(delta.gid);
            if (it_ghost != ghost_nodes.global_to_local.end()) {
                int lid = it_ghost->second;
                local_graph.vertex_labels[lid] = delta.new_label;
                labels_new[lid] = delta.new_label;
            }
        }

        // Step6: Edge-cut 변화율 검사
        int curr_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels);
        double delta = (prev_edge_cut > 0)
                           ? std::abs((double)(curr_edge_cut - prev_edge_cut) / prev_edge_cut)
                           : 1.0;
        
        // Edge-cut 디버깅
        if (mpi_rank == 0 && iter < 3) {
            std::cout << "  [DEBUG] Edge-cut 변화: " << prev_edge_cut << " → " << curr_edge_cut << "\n";
        }
        
        // 반복 결과 출력
        if (mpi_rank == 0) {
            std::cout << "=== Iter " << std::setw(3) << iter + 1 << " ===\n";
            std::cout << "  Edge-cut: " << std::setw(8) << curr_edge_cut 
                      << " (prev: " << std::setw(8) << prev_edge_cut << ")\n";
            std::cout << "  Delta: " << std::setw(8) << std::fixed << std::setprecision(5) << delta * 100 << "%";
            
            if (delta < epsilon) {
                stable_count++;
                std::cout << " [안정화 " << stable_count << "/" << k_limit << "]";
            } else {
                stable_count = 0;
                std::cout << " [변화중]";
            }
            std::cout << "\n";
            
            // 파티션 밸런스 정보
            double max_rv = 0.0, max_re = 0.0;
            for (const auto &p : PI) {
                max_rv = std::max(max_rv, p.RV);
                max_re = std::max(max_re, p.RE);
            }
            std::cout << "  Balance - Vertex: " << std::setw(6) << std::setprecision(3) << max_rv 
                      << ", Edge: " << std::setw(6) << std::setprecision(3) << max_re << "\n";
            
            // 경계 노드 수
            std::cout << "  Boundary nodes: " << boundary_nodes_gid.size() << "\n";
            std::cout << std::endl;
        }
        prev_edge_cut = curr_edge_cut;
        if (stable_count >= k_limit) {
            if (mpi_rank == 0) std::cout << "수렴 완료!\n";
            break;
        }
    }

    auto t_phase2_end = std::chrono::high_resolution_clock::now();
    long exec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_phase2_end - t_phase2_start).count();

    // Balance 계산
    double max_vertex_ratio = 0.0, max_edge_ratio = 0.0;
    for (const auto &p : PI) {
        max_vertex_ratio = std::max(max_vertex_ratio, p.RV);
        max_edge_ratio = std::max(max_edge_ratio, p.RE);
    }
    double avg_vertex_ratio = std::accumulate(PI.begin(), PI.end(), 0.0, [](double acc, const PartitionInfo &p) { return acc + p.RV; }) / num_partitions;
    double avg_edge_ratio = std::accumulate(PI.begin(), PI.end(), 0.0, [](double acc, const PartitionInfo &p) { return acc + p.RE; }) / num_partitions;

    PartitioningMetrics m2;
    m2.edge_cut = prev_edge_cut;
    m2.vertex_balance = max_vertex_ratio / avg_vertex_ratio;
    m2.edge_balance = max_edge_ratio / avg_edge_ratio;
    m2.loading_time_ms = exec_ms;
    m2.distribution_time_ms = 0;
    m2.num_partitions = num_partitions;
    return m2;
}