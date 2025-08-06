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

    // 각 파티션의 노드 수 계산 (owned 노드만)
    for (int u = 0; u < g.num_vertices; u++) {
        int label = labels[u];
        if (label >= 0 && label < num_partitions) {
            local_vertex_counts[label]++;
        }
    }
    
    // 각 파티션의 간선 수 계산 (파티션 내부 간선만)
    for (int u = 0; u < g.num_vertices; u++) {
        int label_u = labels[u];
        if (label_u < 0 || label_u >= num_partitions) continue;
        
        for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; e++) {
            int v = g.col_indices[e];
            int label_v = (v < (int)labels.size()) ? labels[v] : -1;
            if (label_v >= 0 && label_v < num_partitions && label_u == label_v) {
                local_edge_counts[label_u]++;
            }
        }
    }

    // 전역 집계
    std::vector<int> global_vertex_counts(num_partitions, 0);
    std::vector<int> global_edge_counts(num_partitions, 0);

    MPI_Allreduce(local_vertex_counts.data(), global_vertex_counts.data(),
                  num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_edge_counts.data(), global_edge_counts.data(),
                  num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // 전체 그래프 크기 계산
    int total_vertices = std::accumulate(global_vertex_counts.begin(), global_vertex_counts.end(), 0);
    int total_edges = std::accumulate(global_edge_counts.begin(), global_edge_counts.end(), 0);

    // 균등 분배 기준값
    double expected_vertices = static_cast<double>(total_vertices) / num_partitions;
    double expected_edges = (total_edges > 0) ? static_cast<double>(total_edges) / num_partitions : 1.0;

    // 비율 계산
    for (int i = 0; i < num_partitions; i++) {
        PI[i].RV = (expected_vertices > 0) ? static_cast<double>(global_vertex_counts[i]) / expected_vertices : 1.0;
        PI[i].RE = (expected_edges > 0) ? static_cast<double>(global_edge_counts[i]) / expected_edges : 1.0;
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

// 경계 노드를 찾는 함수 (병합된 CSR에서 다른 파티션과 인접한 노드 찾기)
static std::vector<int> extractBoundaryLocalIDs(const Graph &local_graph, const GhostNodes &ghost_nodes, int mpi_rank)
{
    std::vector<int> boundary_nodes;
    
    for (int u = 0; u < local_graph.num_vertices; u++) {
        int u_label = local_graph.vertex_labels[u];
        bool is_boundary = false;
        
        // u의 이웃들을 검사
        for (int edge_idx = local_graph.row_ptr[u]; edge_idx < local_graph.row_ptr[u + 1]; edge_idx++) {
            int v = local_graph.col_indices[edge_idx];
            int v_label;
            
            if (v < local_graph.num_vertices) {
                // local 노드 - 같은 프로세스 내의 다른 파티션일 수 있음
                v_label = local_graph.vertex_labels[v];
            } else {
                // ghost 노드 - 다른 프로세스의 노드
                int ghost_idx = v - local_graph.num_vertices;
                if (ghost_idx < (int)ghost_nodes.ghost_labels.size()) {
                    v_label = ghost_nodes.ghost_labels[ghost_idx];
                } else {
                    continue; // 잘못된 인덱스는 건너뛰기
                }
            }
            
            // 다른 파티션 라벨을 가진 이웃이 있으면 경계 노드
            if (u_label != v_label) {
                is_boundary = true;
                break;
            }
        }
        
        if (is_boundary) {
            boundary_nodes.push_back(u);
        }
    }
    
    return boundary_nodes;
}

// === Edge-cut 계산 ===
static int computeEdgeCut(const Graph &g, const std::vector<int> &labels, const GhostNodes &ghost_nodes)
{
    int local_cut = 0;
    int total_edges = 0;
    
    // owned 노드의 간선만 카운트 (중복 방지)
    for (int u = 0; u < g.num_vertices; u++) {
        for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; e++) {
            int v = g.col_indices[e];
            total_edges++;
            
            // u의 라벨 (owned 노드)
            int u_label = (u < (int)labels.size()) ? labels[u] : -1;
            
            // v의 라벨 결정
            int v_label = -1;
            if (v < g.num_vertices) {
                // local 노드
                v_label = (v < (int)labels.size()) ? labels[v] : -1;
            } else {
                // ghost 노드
                int ghost_idx = v - g.num_vertices;
                if (ghost_idx >= 0 && ghost_idx < (int)ghost_nodes.ghost_labels.size()) {
                    v_label = ghost_nodes.ghost_labels[ghost_idx];
                }
            }
            
            if (u_label != -1 && v_label != -1 && u_label != v_label) {
                local_cut++;
            }
        }
    }
    
    int global_cut = 0;
    int global_total_edges = 0;
    MPI_Allreduce(&local_cut, &global_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_edges, &global_total_edges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // 디버깅을 위한 정보 출력 (첫 번째 호출만)
    static bool first_call = true;
    if (first_call) {
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        if (mpi_rank == 0) {
            std::cout << "  [DEBUG] Initial edge-cut: " << global_cut << " (total_edges: " << global_total_edges << ")\n";
            
            // 첫 번째 몇개 노드의 라벨 샘플 출력
            std::cout << "    Initial node labels: ";
            for (int i = 0; i < std::min(10, g.num_vertices); i++) {
                std::cout << "Node" << i << "(L" << labels[i] << ") ";
            }
            std::cout << "\n";
        }
        first_call = false;
    }
    
    // 분산 환경에서는 각 owned 노드의 간선만 카운트하므로 중복이 없음
    return global_cut;
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

    int prev_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels, ghost_nodes);
    int convergence_count = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        // labels_new를 현재 라벨로 동기화
        labels_new = local_graph.vertex_labels;
        
        // Step1: RV, RE 계산
        calculatePartitionRatios(local_graph, local_graph.vertex_labels, num_partitions, PI);

        // Penalty 계산 및 강화
        calculatePenalty(PI, num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            // penalty 값을 50배 강화하여 라벨 변경을 촉진
            penalty[i] = PI[i].P_L * 50.0;
        }

        // Step3: Boundary 노드 추출(local id)
        auto boundary_nodes_local = extractBoundaryLocalIDs(local_graph, ghost_nodes, mpi_rank);
        
        if (mpi_rank == 0 && iter < 5) {  // 처음 5번만 상세 출력
            std::cout << "  Iter " << iter + 1 << ": Found " << boundary_nodes_local.size() << " boundary nodes\n";
            
            // 파티션별 상태 출력 (처음만)
            if (iter == 0) {
                for (int i = 0; i < num_partitions; i++) {
                    std::cout << "    Partition " << i << ": RV=" << std::fixed << std::setprecision(3) << PI[i].RV 
                              << ", RE=" << PI[i].RE << ", Penalty=" << std::setprecision(6) << PI[i].P_L << "\n";
                }
            }
        }
        
        if (boundary_nodes_local.empty()) {
            if (mpi_rank == 0) std::cout << "경계 노드 없음, 종료\n";
            break;
        }

        // Step4: GPU 커널 실행
        runBoundaryLPOnGPU_Warp(local_graph.row_ptr,
                                local_graph.col_indices,
                                local_graph.vertex_labels, // old labels
                                labels_new,                // new labels
                                penalty,
                                boundary_nodes_local,
                                num_partitions);

        // Step4b: GPU 결과를 실제 라벨 배열에 적용
        int owned_changes = 0;
        for (int lid : boundary_nodes_local) {
            if (lid >= 0 && lid < local_graph.num_vertices && lid < (int)labels_new.size()) {
                if (local_graph.vertex_labels[lid] != labels_new[lid]) {
                    local_graph.vertex_labels[lid] = labels_new[lid];
                    owned_changes++;
                }
            }
        }
        
        if (mpi_rank == 0 && owned_changes > 0) {
            std::cout << "  [DEBUG] GPU updated " << owned_changes << " boundary nodes\n";
        }

        // Step5: 변경된 라벨 정보만 전송 (Delta 구조체 사용)
        std::vector<Delta> delta_changes;
        for (int lid : boundary_nodes_local) {
            // local index가 유효한 범위인지 확인
            if (lid < 0 || lid >= (int)local_graph.global_ids.size()) {
                if (mpi_rank == 0) {
                    std::cout << "  [ERROR] Invalid local index: " << lid << " (max: " << local_graph.global_ids.size() << ")\n";
                }
                continue;
            }
            
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
        int ghost_updates = 0;
        for (const auto &delta : recv_deltas) {
            // ghost 노드인지 확인
            auto it_ghost = ghost_nodes.global_to_local.find(delta.gid);
            if (it_ghost != ghost_nodes.global_to_local.end()) {
                int ghost_idx = it_ghost->second;
                int ghost_lid = local_graph.num_vertices + ghost_idx; // 올바른 ghost 노드 인덱스
                
                if (ghost_lid < (int)local_graph.vertex_labels.size()) {
                    if (local_graph.vertex_labels[ghost_lid] != delta.new_label) {
                        ghost_updates++;
                    }
                    local_graph.vertex_labels[ghost_lid] = delta.new_label;
                    labels_new[ghost_lid] = delta.new_label;
                    
                    // Ghost 노드 배열도 업데이트
                    if (ghost_idx < (int)ghost_nodes.ghost_labels.size()) {
                        ghost_nodes.ghost_labels[ghost_idx] = delta.new_label;
                    }
                }
            }
        }

        // Step6: Edge-cut 변화율 검사
        int curr_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels, ghost_nodes);
        double delta = (prev_edge_cut > 0)
                           ? std::abs((double)(curr_edge_cut - prev_edge_cut) / prev_edge_cut)
                           : 1.0;
        
        // 수렴 조건 확인
        if (delta < epsilon) {
            convergence_count++;
        } else {
            convergence_count = 0; // 리셋
        }
        
        // 반복 결과 출력
        if (mpi_rank == 0) {
            std::cout << "Iter " << iter + 1 << ": Edge-cut " << curr_edge_cut 
                      << " (delta: " << std::fixed << std::setprecision(3) << delta * 100 << "%)\n";
        }
        prev_edge_cut = curr_edge_cut;
        
        if (convergence_count >= k_limit) {
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
    
    // 총 노드 수 계산 (owned 노드만)
    int local_vertices = local_graph.num_vertices;
    int global_vertices = 0;
    MPI_Allreduce(&local_vertices, &global_vertices, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    m2.total_vertices = global_vertices;
    
    // 총 간선 수 계산 (전체 그래프의 실제 간선 수)
    int local_edges = 0;
    for (int u = 0; u < local_graph.num_vertices; u++) {
        local_edges += (local_graph.row_ptr[u + 1] - local_graph.row_ptr[u]);
    }
    int global_edges = 0;
    MPI_Allreduce(&local_edges, &global_edges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    m2.total_edges = global_edges; // 이것이 전체 그래프의 실제 간선 수
    
    return m2;
}