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

#include "graph_types.h"
#include "phase2/phase2.h"
#include "phase2/gpu_lp_boundary.h"

// === 공통 함수들 ===

// Ghost 노드 라벨 안전하게 가져오는 인라인 함수
inline int getNodeLabel(int node_id, const Graph &g, const std::vector<int> &labels, 
                       const GhostNodes &ghost_nodes) {
    if (node_id < g.num_vertices) {
        return labels[node_id];
    } else {
        int ghost_idx = node_id - g.num_vertices;
        return (ghost_idx >= 0 && ghost_idx < (int)ghost_nodes.ghost_labels.size()) 
               ? ghost_nodes.ghost_labels[ghost_idx] : -1;
    }
}

// 파티션별 통계 계산 (최적화된 버전 - 한 번의 MPI 호출로 통합)
static PartitionStats computePartitionStats(const Graph &g, const std::vector<int> &labels, 
                                           const GhostNodes &ghost_nodes, int num_partitions) {
    PartitionStats stats;
    stats.local_vertex_counts.resize(num_partitions, 0);
    stats.local_edge_counts.resize(num_partitions, 0);
    stats.global_vertex_counts.resize(num_partitions, 0);
    stats.global_edge_counts.resize(num_partitions, 0);

    // 각 파티션의 노드 수 계산 (owned 노드만) - OpenMP 병렬화
    #pragma omp parallel
    {
        std::vector<int> thread_vertex_counts(num_partitions, 0);
        
        #pragma omp for nowait
        for (int u = 0; u < g.num_vertices; u++) {
            int label = labels[u];
            if (label >= 0 && label < num_partitions) {
                thread_vertex_counts[label]++;
            }
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < num_partitions; i++) {
                stats.local_vertex_counts[i] += thread_vertex_counts[i];
            }
        }
    }
    
    // 각 파티션의 간선 수 계산 (파티션 내부 간선만) - OpenMP 병렬화
    #pragma omp parallel
    {
        std::vector<int> thread_edge_counts(num_partitions, 0);
        
        #pragma omp for nowait
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
            for (int i = 0; i < num_partitions; i++) {
                stats.local_edge_counts[i] += thread_edge_counts[i];
            }
        }
    }

    // 💡 최적화: 단일 MPI 호출로 통합 (vertex + edge counts 동시 전송)
    std::vector<int> send_buffer(2 * num_partitions);
    std::vector<int> recv_buffer(2 * num_partitions);
    
    // 버퍼 패킹: [vertex_counts..., edge_counts...]
    std::copy(stats.local_vertex_counts.begin(), stats.local_vertex_counts.end(), send_buffer.begin());
    std::copy(stats.local_edge_counts.begin(), stats.local_edge_counts.end(), send_buffer.begin() + num_partitions);
    
    // 단일 Allreduce로 모든 카운트 집계
    MPI_Allreduce(send_buffer.data(), recv_buffer.data(), 2 * num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // 결과 언패킹
    std::copy(recv_buffer.begin(), recv_buffer.begin() + num_partitions, stats.global_vertex_counts.begin());
    std::copy(recv_buffer.begin() + num_partitions, recv_buffer.end(), stats.global_edge_counts.begin());

    // 전체 그래프 크기 계산
    stats.total_vertices = std::accumulate(stats.global_vertex_counts.begin(), stats.global_vertex_counts.end(), 0);
    stats.total_edges = std::accumulate(stats.global_edge_counts.begin(), stats.global_edge_counts.end(), 0);

    // 균등 분배 기준값
    stats.expected_vertices = static_cast<double>(stats.total_vertices) / num_partitions;
    stats.expected_edges = (stats.total_edges > 0) ? static_cast<double>(stats.total_edges) / num_partitions : 1.0;

    return stats;
}

// MPI Delta 통신 헬퍼 함수
static std::vector<Delta> allgatherDeltas(const std::vector<Delta> &local_deltas, int mpi_size) {
    int send_count = local_deltas.size();
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
    MPI_Type_contiguous(2, MPI_INT, &MPI_DELTA);
    MPI_Type_commit(&MPI_DELTA);

    MPI_Allgatherv(local_deltas.data(), send_count, MPI_DELTA,
                   recv_deltas.data(), recv_counts.data(), displs.data(),
                   MPI_DELTA, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_DELTA);
    
    return recv_deltas;
}

// === Penalty 계산 방식 선택 (실험용) ===
// #define USE_MASTER_WORKER_PENALTY  // 이 줄을 주석 해제하면 Master-Worker 방식 사용

#ifdef USE_MASTER_WORKER_PENALTY
// Master-Worker 방식: Rank 0만 계산, 결과를 브로드캐스트
std::vector<double> calculatePenalties(
    const PartitionStats &stats,
    int num_partitions,
    int mpi_rank = 0)
{
    std::vector<double> penalties(num_partitions);
    
    // Rank 0만 penalty 계산 수행
    if (mpi_rank == 0) {
        // RV, RE 비율 계산
        std::vector<double> RV(num_partitions), RE(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            RV[i] = (stats.expected_vertices > 0) ? static_cast<double>(stats.global_vertex_counts[i]) / stats.expected_vertices : 1.0;
            RE[i] = (stats.expected_edges > 0) ? static_cast<double>(stats.global_edge_counts[i]) / stats.expected_edges : 1.0;
        }
        
        // 디버깅 출력 (Rank 0만)
        printf("\n=== Label Statistics (Master-Worker 방식) ===\n");
        printf("Total: %d vertices, %d edges\n", stats.total_vertices, stats.total_edges);
        printf("Expected per label: %.1f vertices, %.1f edges\n", stats.expected_vertices, stats.expected_edges);
        
        for (int i = 0; i < num_partitions; i++) {
            printf("Label %d: %d vertices (%.3f), %d edges (%.3f)\n", 
                   i, stats.global_vertex_counts[i], RV[i], 
                   stats.global_edge_counts[i], RE[i]);
        }
        printf("========================\n\n");

        // Penalty 직접 계산
        double rv_mean = 0.0, re_mean = 0.0;
        for (int i = 0; i < num_partitions; i++) {
            rv_mean += RV[i];
            re_mean += RE[i];
        }
        rv_mean /= num_partitions;
        re_mean /= num_partitions;

        double rv_var = 0.0, re_var = 0.0;
        for (int i = 0; i < num_partitions; i++) {
            rv_var += (RV[i] - rv_mean) * (RV[i] - rv_mean);
            re_var += (RE[i] - re_mean) * (RE[i] - re_mean);
        }
        rv_var /= num_partitions;
        re_var /= num_partitions;

        double total_var = rv_var + re_var;
        double imb_rv = (total_var > 0) ? rv_var / total_var : 0.0;
        double imb_re = (total_var > 0) ? re_var / total_var : 0.0;

        // Penalty 배열 직접 생성
        for (int i = 0; i < num_partitions; i++) {
            double G_RV = (1.0 - RV[i]) / num_partitions;
            double G_RE = (1.0 - RE[i]) / num_partitions;
            penalties[i] = imb_rv * G_RV + imb_re * G_RE;
        }
    }
    
    // 결과를 모든 프로세서에 브로드캐스트
    MPI_Bcast(penalties.data(), num_partitions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return penalties;
}

#else
// 기존 방식: 모든 프로세서가 동일한 계산 수행
std::vector<double> calculatePenalties(
    const PartitionStats &stats,
    int num_partitions,
    int mpi_rank = 0)
{
    // RV, RE 비율 계산
    std::vector<double> RV(num_partitions), RE(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        RV[i] = (stats.expected_vertices > 0) ? static_cast<double>(stats.global_vertex_counts[i]) / stats.expected_vertices : 1.0;
        RE[i] = (stats.expected_edges > 0) ? static_cast<double>(stats.global_edge_counts[i]) / stats.expected_edges : 1.0;
    }
    
    // 디버깅 출력 (Rank 0만)
    if (mpi_rank == 0) {
        printf("\n=== Label Statistics (기존 방식) ===\n");
        printf("Total: %d vertices, %d edges\n", stats.total_vertices, stats.total_edges);
        printf("Expected per label: %.1f vertices, %.1f edges\n", stats.expected_vertices, stats.expected_edges);
        
        for (int i = 0; i < num_partitions; i++) {
            printf("Label %d: %d vertices (%.3f), %d edges (%.3f)\n", 
                   i, stats.global_vertex_counts[i], RV[i], 
                   stats.global_edge_counts[i], RE[i]);
        }
        printf("========================\n\n");
    }

    // Penalty 직접 계산
    double rv_mean = 0.0, re_mean = 0.0;
    for (int i = 0; i < num_partitions; i++) {
        rv_mean += RV[i];
        re_mean += RE[i];
    }
    rv_mean /= num_partitions;
    re_mean /= num_partitions;

    double rv_var = 0.0, re_var = 0.0;
    for (int i = 0; i < num_partitions; i++) {
        rv_var += (RV[i] - rv_mean) * (RV[i] - rv_mean);
        re_var += (RE[i] - re_mean) * (RE[i] - re_mean);
    }
    rv_var /= num_partitions;
    re_var /= num_partitions;

    double total_var = rv_var + re_var;
    double imb_rv = (total_var > 0) ? rv_var / total_var : 0.0;
    double imb_re = (total_var > 0) ? re_var / total_var : 0.0;

    // Penalty 배열 직접 생성
    std::vector<double> penalties(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        double G_RV = (1.0 - RV[i]) / num_partitions;
        double G_RE = (1.0 - RE[i]) / num_partitions;
        penalties[i] = imb_rv * G_RV + imb_re * G_RE;
    }

    return penalties;
}
#endif

// 경계 노드를 찾는 함수 (최적화된 버전) - OpenMP 병렬화
static std::vector<int> extractBoundaryLocalIDs(const Graph &local_graph, const GhostNodes &ghost_nodes)
{
    std::vector<int> boundary_nodes;
    
    #pragma omp parallel
    {
        std::vector<int> thread_boundary_nodes;
        
        #pragma omp for nowait
        for (int u = 0; u < local_graph.num_vertices; u++) {
            int u_label = local_graph.vertex_labels[u];
            bool is_boundary = false;
            
            // u의 이웃들을 검사
            for (int edge_idx = local_graph.row_ptr[u]; edge_idx < local_graph.row_ptr[u + 1]; edge_idx++) {
                int v = local_graph.col_indices[edge_idx];
                int v_label = getNodeLabel(v, local_graph, local_graph.vertex_labels, ghost_nodes);
                
                // 다른 파티션 라벨을 가진 이웃이 있으면 경계 노드
                if (v_label != -1 && u_label != v_label) {
                    is_boundary = true;
                    break;
                }
            }
            
            if (is_boundary) {
                thread_boundary_nodes.push_back(u);
            }
        }
        
        #pragma omp critical
        {
            boundary_nodes.insert(boundary_nodes.end(), 
                                thread_boundary_nodes.begin(), 
                                thread_boundary_nodes.end());
        }
    }
    
    return boundary_nodes;
}

// === Edge-cut 계산 (최적화된 버전) === - OpenMP 병렬화
static int computeEdgeCut(const Graph &g, const std::vector<int> &labels, const GhostNodes &ghost_nodes)
{
    int local_cut = 0;
    int total_edges = 0;
    
    // owned 노드의 간선만 카운트 (중복 방지) - OpenMP 병렬화
    #pragma omp parallel reduction(+:local_cut,total_edges)
    {
        #pragma omp for nowait
        for (int u = 0; u < g.num_vertices; u++) {
            for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; e++) {
                int v = g.col_indices[e];
                total_edges++;
                
                // u의 라벨 (owned 노드만 처리하므로 항상 유효)
                int u_label = labels[u];
                
                // v의 라벨 결정 (최적화된 함수 사용)
                int v_label = getNodeLabel(v, g, labels, ghost_nodes);
                
                // 다른 파티션 간 간선이면 edge-cut에 포함
                if (u_label != -1 && v_label != -1 && u_label != v_label) {
                    local_cut++;
                }
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
    const double epsilon = 0.03; // 수렴 기준
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
    
    // CPU 메모리 Pin 최적화: 자주 사용되는 벡터들을 Pinned Memory로 할당
    std::vector<int> labels_new;
    std::vector<double> penalty_pinned;
    std::vector<int> boundary_nodes_pinned;
    
    // GPU와 자주 통신하는 메모리를 Pinned로 할당 (성능 향상)
    labels_new.resize(local_graph.vertex_labels.size());
    penalty_pinned.reserve(num_partitions);  // 미리 공간 확보
    
    // 모든 프로세스 동기화
    MPI_Barrier(MPI_COMM_WORLD);

    // *** CRITICAL: Phase1과 Phase2 사이의 Ghost Node 라벨 동기화 ***
    // Phase1 완료 후 모든 owned 노드의 최신 라벨을 다른 프로세스와 공유
    std::vector<Delta> all_owned_labels;
    for (int i = 0; i < local_graph.num_vertices; i++) {
        if (i < (int)local_graph.global_ids.size()) {
            Delta delta;
            delta.gid = local_graph.global_ids[i];
            delta.new_label = local_graph.vertex_labels[i];
            all_owned_labels.push_back(delta);
        }
    }
    
    // 최적화된 Delta 통신 사용
    std::vector<Delta> recv_all_labels = allgatherDeltas(all_owned_labels, mpi_size);
    
    // Ghost 노드 라벨 업데이트
    for (const auto &delta : recv_all_labels) {
        auto it_ghost = ghost_nodes.global_to_local.find(delta.gid);
        if (it_ghost != ghost_nodes.global_to_local.end()) {
            int ghost_idx = it_ghost->second;
            if (ghost_idx >= 0 && ghost_idx < (int)ghost_nodes.ghost_labels.size()) {
                ghost_nodes.ghost_labels[ghost_idx] = delta.new_label;
                
                // local_graph의 vertex_labels도 동기화
                int ghost_lid = local_graph.num_vertices + ghost_idx;
                if (ghost_lid < (int)local_graph.vertex_labels.size()) {
                    local_graph.vertex_labels[ghost_lid] = delta.new_label;
                }
            }
        }
    }
    
    if (mpi_rank == 0) {
        std::cout << "Phase1-Phase2 라벨 동기화 완료 (전체 라벨: " << recv_all_labels.size() << "개)" << std::endl;
    }

    labels_new = local_graph.vertex_labels; // 현재 라벨 복사 (최적화된 할당)

    int prev_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels, ghost_nodes);
    int convergence_count = 0;

    // 메모리 풀 최적화: 파티션 통계를 재사용하여 메모리 할당 최소화
    PartitionStats current_stats = computePartitionStats(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions);
    
    // 메모리 예약: 반복에서 사용될 벡터들 미리 할당
    penalty_pinned.resize(num_partitions);
    std::vector<Delta> delta_changes;
    delta_changes.reserve(1000);  // 예상 변경사항 수만큼 미리 할당

    for (int iter = 0; iter < max_iter; iter++) {
        // 메모리 재사용: labels_new 벡터를 재할당하지 않고 내용만 복사
        std::copy(local_graph.vertex_labels.begin(), local_graph.vertex_labels.end(), labels_new.begin());
        
        // Step1: 최적화된 penalty 계산 (통계 재사용, 메모리 풀에서 할당)
        penalty_pinned = calculatePenalties(current_stats, num_partitions, mpi_rank);

        // Step2: Boundary 노드 추출(local id) - 메모리 풀 재사용
        boundary_nodes_pinned = extractBoundaryLocalIDs(local_graph, ghost_nodes);
        
        if (boundary_nodes_pinned.empty()) {
            if (mpi_rank == 0) std::cout << "경계 노드 없음, 종료\n";
            break;
        }

        // Step3: 고성능 GPU 커널 실행 (다단계 최적화)
#ifdef USE_PINNED_MEMORY_OPTIMIZATION
        // 1순위: Pinned Memory 최적화 (메모리 누수 방지 + 최고 성능)
        try {
            runBoundaryLPOnGPU_PinnedOptimized(local_graph.row_ptr,
                                             local_graph.col_indices,
                                             local_graph.vertex_labels, // old labels
                                             labels_new,                // new labels
                                             penalty_pinned,            // pinned penalty 배열
                                             boundary_nodes_pinned,     // pinned boundary 배열
                                             num_partitions);
        } catch (const std::exception& e) {
            printf("[Rank %d] Pinned 메모리 최적화 실패, Thrust로 전환: %s\n", mpi_rank, e.what());
            
#ifdef USE_THRUST_OPTIMIZATION
            try {
                runBoundaryLPOnGPU_Thrust_Optimized(local_graph.row_ptr,
                                                   local_graph.col_indices,
                                                   local_graph.vertex_labels,
                                                   labels_new,
                                                   penalty_pinned,
                                                   boundary_nodes_pinned,
                                                   num_partitions);
            } catch (const std::exception& e2) {
                printf("[Rank %d] Thrust 최적화도 실패, 기본 버전으로 전환: %s\n", mpi_rank, e2.what());
                runBoundaryLPOnGPU_Optimized(local_graph.row_ptr,
                                           local_graph.col_indices,
                                           local_graph.vertex_labels,
                                           labels_new,
                                           penalty_pinned,
                                           boundary_nodes_pinned,
                                           num_partitions);
            }
#else
            runBoundaryLPOnGPU_Optimized(local_graph.row_ptr,
                                       local_graph.col_indices,
                                       local_graph.vertex_labels,
                                       labels_new,
                                       penalty_pinned,
                                       boundary_nodes_pinned,
                                       num_partitions);
#endif
        }
        
#elif defined(USE_THRUST_OPTIMIZATION)
        // 2순위: Thrust 최적화
        try {
            runBoundaryLPOnGPU_Thrust_Optimized(local_graph.row_ptr,
                                               local_graph.col_indices,
                                               local_graph.vertex_labels, // old labels
                                               labels_new,                // new labels
                                               penalty_pinned,            // pinned penalty 배열
                                               boundary_nodes_pinned,     // pinned boundary 배열
                                               num_partitions);
        } catch (const std::exception& e) {
            printf("[Rank %d] Thrust 최적화 실패, 안전 모드로 전환: %s\n", mpi_rank, e.what());
            runBoundaryLPOnGPU_Safe(local_graph.row_ptr,
                                  local_graph.col_indices,
                                  local_graph.vertex_labels,
                                  labels_new,
                                  penalty_pinned,
                                  boundary_nodes_pinned,
                                  num_partitions);
        }
#else
        // 기본: 표준 최적화
        runBoundaryLPOnGPU_Optimized(local_graph.row_ptr,
                                   local_graph.col_indices,
                                   local_graph.vertex_labels, // old labels
                                   labels_new,                // new labels
                                   penalty_pinned,            // pinned penalty 배열
                                   boundary_nodes_pinned,     // pinned boundary 배열
                                   num_partitions);
#endif
        
        // GPU 동기화
        cudaDeviceSynchronize();

        // Step4b: GPU 결과를 실제 라벨 배열에 적용 & 변경사항을 Delta로 수집
        delta_changes.clear();  // 메모리 재사용: clear()로 용량 유지
        
        for (int lid : boundary_nodes_pinned) {
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

        // Step5: 최적화된 Delta 통신 사용
        std::vector<Delta> recv_deltas = allgatherDeltas(delta_changes, mpi_size);

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
        
        // 다음 반복을 위해 파티션 통계 업데이트 (라벨이 변경된 경우에만)
        if (!delta_changes.empty() || !recv_deltas.empty()) {
            current_stats = computePartitionStats(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions);
        }
    }

    auto t_phase2_end = std::chrono::high_resolution_clock::now();
    long exec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_phase2_end - t_phase2_start).count();

    // GPU 사용 통계 출력
    std::cout << "[Rank " << mpi_rank << "] Phase2 완료 - GPU " << gpu_id 
              << " 총 실행시간: " << exec_ms << "ms" << std::endl;
    
    // 최종 Balance 계산을 위한 통계 (이미 계산된 current_stats 재사용)
    PartitionStats final_stats = computePartitionStats(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions);

    double max_vertex_ratio = 0.0, max_edge_ratio = 0.0;
    double sum_vertex_ratio = 0.0, sum_edge_ratio = 0.0;
    
    for (int i = 0; i < num_partitions; i++) {
        double rv = (final_stats.expected_vertices > 0) ? static_cast<double>(final_stats.global_vertex_counts[i]) / final_stats.expected_vertices : 1.0;
        double re = (final_stats.expected_edges > 0) ? static_cast<double>(final_stats.global_edge_counts[i]) / final_stats.expected_edges : 1.0;
        
        max_vertex_ratio = std::max(max_vertex_ratio, rv);
        max_edge_ratio = std::max(max_edge_ratio, re);
        sum_vertex_ratio += rv;
        sum_edge_ratio += re;
    }
    
    double avg_vertex_ratio = sum_vertex_ratio / num_partitions;
    double avg_edge_ratio = sum_edge_ratio / num_partitions;

    PartitioningMetrics m2;
    m2.edge_cut = prev_edge_cut;
    m2.vertex_balance = max_vertex_ratio / avg_vertex_ratio;
    m2.edge_balance = max_edge_ratio / avg_edge_ratio;
    m2.loading_time_ms = exec_ms;
    m2.distribution_time_ms = 0;
    m2.num_partitions = num_partitions;
    
    // 이미 계산된 통계에서 전역 정보 사용
    m2.total_vertices = final_stats.total_vertices;
    m2.total_edges = final_stats.total_edges;

    MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();  // 모든 GPU 작업 완료 대기
    std::cout << "[Rank " << mpi_rank << "] GPU " << gpu_id << " 작업 완료" << std::endl;
    std::cout.flush();
    
    return m2;
}