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

#include "graph_types.h"
#include "phase2/phase2.h"
#include "phase2/gpu_lp_boundary.h"

// === Partition 비율 계산 ===
void calculatePartitionRatios(
    const Graph &g,
    const std::vector<int> &labels,
    const GhostNodes &ghost_nodes,
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
            int label_v = -1;
            
            if (v < g.num_vertices) {
                // local 노드 - labels 배열에서 직접 가져오기
                label_v = labels[v];
            } else {
                // ghost 노드 - ghost_nodes 구조체에서 가져오기
                int ghost_idx = v - g.num_vertices;
                if (ghost_idx >= 0 && ghost_idx < (int)ghost_nodes.ghost_labels.size()) {
                    label_v = ghost_nodes.ghost_labels[ghost_idx];
                }
            }
            
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

    for (int i = 0; i < num_partitions; i++) {
        auto &p = PI[i];
        p.imb_RV = imb_rv;
        p.imb_RE = imb_re;
        p.G_RV = (1.0 - p.RV) / num_partitions;
        p.G_RE = (1.0 - p.RE) / num_partitions;
        p.P_L = imb_rv * p.G_RV + imb_re * p.G_RE;
    }
}

// 경계 노드를 찾는 함수 (병합된 CSR에서 다른 파티션과 인접한 노드 찾기)
static std::vector<int> extractBoundaryLocalIDs( const Graph &local_graph, const GhostNodes &ghost_nodes)
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
                // local 노드 - vertex_labels에서 직접 가져오기
                v_label = local_graph.vertex_labels[v];
            } else {
                // ghost 노드 - ghost_nodes 구조체에서 가져오기
                int ghost_idx = v - local_graph.num_vertices;
                if (ghost_idx >= 0 && ghost_idx < (int)ghost_nodes.ghost_labels.size()) {
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
            
            // u의 라벨 (owned 노드만 처리하므로 항상 유효)
            int u_label = labels[u];
            
            // v의 라벨 결정
            int v_label = -1;
            if (v < g.num_vertices) {
                // local 노드 - labels 배열에서 직접 가져오기
                v_label = labels[v];
            } else {
                // ghost 노드 - ghost_nodes 구조체에서 가져오기
                int ghost_idx = v - g.num_vertices;
                if (ghost_idx >= 0 && ghost_idx < (int)ghost_nodes.ghost_labels.size()) {
                    v_label = ghost_nodes.ghost_labels[ghost_idx];
                }
            }
            
            // 다른 파티션 간 간선이면 edge-cut에 포함
            if (u_label != -1 && v_label != -1 && u_label != v_label) {
                local_cut++;
            }
        }
    }
    
    int global_cut = 0;
    int global_total_edges = 0;
    MPI_Allreduce(&local_cut, &global_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_edges, &global_total_edges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // 분산 환경에서는 각 owned 노드의 간선만 카운트하므로 중복이 없음
    return global_cut;
}

// === Phase2 실행 ===
PartitioningMetrics run_phase2(
    int mpi_rank, int mpi_size,
    int num_partitions,
    Graph &local_graph,
    GhostNodes &ghost_nodes,
    int gpu_id)
{
    const int max_iter = 500;
    const double epsilon = 0.01; // 수렴 기준
    const int k_limit = 10;

    auto t_phase2_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "[Rank " << mpi_rank << "/" << mpi_size << "] Phase2 시작 (GPU " << gpu_id << ")" << std::endl;
    std::cout.flush();
    
    // GPU가 이미 할당되었으므로 추가 설정만 확인
    int current_device;
    cudaGetDevice(&current_device);
    if (current_device != gpu_id) {
        cudaSetDevice(gpu_id);
        std::cout << "[Rank " << mpi_rank << "] GPU " << gpu_id << " 재설정 완료" << std::endl;
    }
    
    // 모든 프로세스 동기화
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> labels_new = local_graph.vertex_labels; // 현재 라벨 복사
    std::vector<PartitionInfo> PI(num_partitions);

    int prev_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels, ghost_nodes);
    int convergence_count = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        // labels_new를 현재 라벨로 동기화
        labels_new = local_graph.vertex_labels;
        
        // Step1: RV, RE 계산
        calculatePartitionRatios(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions, PI);

        // Penalty 계산
        calculatePenalty(PI, num_partitions);
        
        // 디버깅: 패널티 값 출력 (첫 번째 iteration만)
        if (iter == 0 && mpi_rank == 0) {
            std::cout << "[Debug] RV values: ";
            for (int i = 0; i < num_partitions; i++) {
                std::cout << "P" << i << "=" << std::fixed << std::setprecision(4) << PI[i].RV << " ";
            }
            std::cout << std::endl;
            
            std::cout << "[Debug] RE values: ";
            for (int i = 0; i < num_partitions; i++) {
                std::cout << "P" << i << "=" << std::fixed << std::setprecision(4) << PI[i].RE << " ";
            }
            std::cout << std::endl;
            
            std::cout << "[Debug] Penalty values: ";
            for (int i = 0; i < num_partitions; i++) {
                std::cout << "P" << i << "=" << std::fixed << std::setprecision(6) << PI[i].P_L << " ";
            }
            std::cout << std::endl;
            
            std::cout << "[Debug] imb_RV=" << std::fixed << std::setprecision(6) << PI[0].imb_RV 
                      << ", imb_RE=" << PI[0].imb_RE << std::endl;
        }

        // Step3: Boundary 노드 추출(local id)
        auto boundary_nodes_local = extractBoundaryLocalIDs(local_graph, ghost_nodes);
        
        // 디버깅: 경계 노드 정보 출력
        std::cout << "[Rank " << mpi_rank << "] Boundary nodes: " << boundary_nodes_local.size() << std::endl;
        if (boundary_nodes_local.size() > 0 && boundary_nodes_local.size() <= 10) {
            std::cout << "[Rank " << mpi_rank << "] First few boundary nodes: ";
            for (int i = 0; i < std::min(5, (int)boundary_nodes_local.size()); i++) {
                std::cout << boundary_nodes_local[i] << " ";
            }
            std::cout << std::endl;
        }
        
        if (boundary_nodes_local.empty()) {
            if (mpi_rank == 0) std::cout << "경계 노드 없음, 종료\n";
            break;
        }

        // Step4: GPU 커널 실행 (Warp 최적화 버전 사용)
        bool enable_adaptive_scaling = true;  // 적응적 스케일링 활성화
        
        // PartitionInfo에서 penalty 배열 추출
        std::vector<double> penalty(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            penalty[i] = PI[i].P_L;
        }
        
        // 패널티 값이 너무 작으면 증폭
        double max_penalty = *std::max_element(penalty.begin(), penalty.end());
        double min_penalty = *std::min_element(penalty.begin(), penalty.end());
        double penalty_range = max_penalty - min_penalty;
        
        if (penalty_range < 0.01) {  // 패널티 차이가 1% 미만이면 10배 증폭
            for (int i = 0; i < num_partitions; i++) {
                penalty[i] *= 10.0;
            }
            if (iter == 0 && mpi_rank == 0) {
                std::cout << "[Debug] Penalty amplified by 10x due to small range (" 
                          << std::fixed << std::setprecision(6) << penalty_range << ")" << std::endl;
            }
        }
        
        runBoundaryLPOnGPU_Warp(local_graph.row_ptr,
                                local_graph.col_indices,
                                local_graph.vertex_labels, // old labels
                                labels_new,                // new labels
                                penalty,                   // penalty 배열 전달
                                boundary_nodes_local,
                                num_partitions,
                                enable_adaptive_scaling);
        
        // GPU 동기화
        cudaDeviceSynchronize();

        // Step4b: GPU 결과를 실제 라벨 배열에 적용 & 변경사항을 Delta로 수집
        std::vector<Delta> delta_changes;
        
        for (int lid : boundary_nodes_local) {
            if (lid >= 0 && lid < local_graph.num_vertices && lid < (int)labels_new.size()) {
                if (local_graph.vertex_labels[lid] != labels_new[lid]) {
                    // Delta 구조체에 변경사항 기록 (적용 전 상태)
                    if (lid < (int)local_graph.global_ids.size()) {
                        Delta delta;
                        delta.gid = local_graph.global_ids[lid];
                        delta.new_label = labels_new[lid];
                        delta_changes.push_back(delta);
                    }
                    
                    // 실제 라벨 적용
                    local_graph.vertex_labels[lid] = labels_new[lid];
                }
            }
        }
        
        std::cout << "[Rank " << mpi_rank << "] Label changes: " << delta_changes.size() << std::endl;

        // Step5: 변경된 라벨 정보만 전송 (Delta 구조체 사용)

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
                int ghost_idx = it_ghost->second;
                
                // 올바른 범위 확인
                if (ghost_idx >= 0 && ghost_idx < (int)ghost_nodes.ghost_labels.size()) {
                    // Ghost 노드 구조체 업데이트 (primary source)
                    ghost_nodes.ghost_labels[ghost_idx] = delta.new_label;
                    
                    // local_graph의 vertex_labels도 동기화 (ghost 노드 부분)
                    int ghost_lid = local_graph.num_vertices + ghost_idx;
                    if (ghost_lid < (int)local_graph.vertex_labels.size()) {
                        local_graph.vertex_labels[ghost_lid] = delta.new_label;
                    }
                }
            }
        }

        // Step6: Edge-cut 변화율 검사
        int curr_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels, ghost_nodes);
        double delta = (prev_edge_cut > 0)
                           ? std::abs((double)(curr_edge_cut - prev_edge_cut) / prev_edge_cut)
                           : 1.0;
        
        // 수렴 조건 확인 (edge-cut 변화율이 epsilon 미만일 때)
        if (delta < epsilon) {
            convergence_count++;
        } else {
            convergence_count = 0; // 리셋
        }
        
        // 반복 결과 출력
        if (mpi_rank == 0) {
            std::cout << "Iter " << iter + 1 << ": Edge-cut " << curr_edge_cut 
                      << " (delta: " << std::fixed << std::setprecision(3) << delta * 100 << "%)";
            
            if (convergence_count > 0) {
                std::cout << " [수렴 카운트: " << convergence_count << "/" << k_limit << "]";
            }
            std::cout << "\n";
        }
        prev_edge_cut = curr_edge_cut;
        
        // 수렴 완료 조건: edge-cut 변화율이 epsilon 미만으로 k_limit 번 연속 발생
        if (convergence_count >= k_limit) {
            if (mpi_rank == 0) {
                std::cout << "수렴 완료! (연속 " << k_limit << "회 변화율 < " 
                          << std::fixed << std::setprecision(1) << epsilon * 100 << "%)\n";
            }
            break;
        }
    }

    auto t_phase2_end = std::chrono::high_resolution_clock::now();
    long exec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_phase2_end - t_phase2_start).count();

    // GPU 사용 통계 출력
    std::cout << "[Rank " << mpi_rank << "] Phase2 완료 - GPU " << gpu_id 
              << " 총 실행시간: " << exec_ms << "ms" << std::endl;
    
    // GPU 메모리 정리
    cudaDeviceReset();
    std::cout << "[Rank " << mpi_rank << "] GPU " << gpu_id << " 리셋 완료" << std::endl;
    std::cout.flush();

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