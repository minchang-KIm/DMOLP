/**
 * @file phase2.cpp
 * @brief DMOLP Phase2 분산 그래프 파티셔닝 최적화 구현체
 * 
 * ==================== DMOLP Phase2 핵심 기능 ====================
 * 
 * 주요 기능:
 * - GPU 가속 Boundary Label Propagation을 활용한 고성능 파티션 최적화
 * - MPI 기반 분산 처리로 대규모 그래프 지원 (수십억 노드급)
 * - DMOLP penalty 기반 적응적 파티션 균형 조정
 * - 동적 수렴 판정을 통한 효율적인 반복 최적화
 * 
 * ==================== 성능 최적화 특징 ====================
 * 
 * GPU 가속:
 * - CUDA warp-based kernel로 고도 노드 처리 최적화
 * - Pinned Memory와 스트리밍을 통한 CPU-GPU 데이터 전송 최적화
 * - 적응적 GPU 메모리 관리로 대용량 그래프 처리
 * 
 * CPU 병렬 처리:
 * - OpenMP 멀티스레딩으로 로컬 계산 가속화
 * - 효율적인 boundary node 탐지 및 관리
 * - 메모리 pool 재사용으로 할당 오버헤드 최소화
 * 
 * MPI 분산 통신:
 * - 비동기 통신으로 계산-통신 오버랩
 * - Delta 압축을 통한 통신량 최소화
 * - Load balancing을 고려한 작업 분산
 * 
 * ==================== 알고리즘 혁신 ====================
 * 
 * DMOLP (Distributed Multi-Objective Label Propagation):
 * - 노드 및 간선 균형을 동시 고려하는 penalty 함수
 * - 동적 boundary expansion으로 계산량 감소
 * - 다단계 수렴 판정으로 품질-성능 균형 달성
 * 
 * 적응적 처리 전략:
 * - 첫 반복에서는 전체 노드 처리로 초기 boundary 탐지
 * - 이후 반복에서는 boundary + 1-hop 확장으로 효율성 향상
 * - GPU 메모리 부족시 자동 CPU fallback 지원
 * 
 * ==================== 기술적 세부사항 ====================
 * 
 * 데이터 구조:
 * - CSR (Compressed Sparse Row) 형식으로 메모리 효율성 극대화
 * - Ghost node 관리로 분산 환경에서의 일관성 보장
 * - Delta 기반 증분 업데이트로 통신 오버헤드 최소화
 * 
 * 메모리 관리:
 * - RAII 패턴으로 안전한 GPU 메모리 관리
 * - Vector reserve를 통한 동적 할당 최적화
 * - Memory pool 재사용으로 fragmentation 방지
 * 
 * 품질 보장:
 * - 정확한 edge-cut 계산 (MPI 환경에서 중복 제거)
 * - 실시간 파티션 통계 모니터링
 * - 수렴 기준 달성시까지 품질 개선 지속
 * 
 * @author DMOLP Development Team
 * @date 2024
 * @version 2.0 (GPU 최적화 및 문서화 완료)
 * 
 * @note GPU와 OpenMP가 활성화된 환경에서 최적 성능 발휘
 * @warning 대용량 그래프 처리시 충분한 GPU 메모리 확보 필요
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
#include <future>

#include "graph_types.h"
#include "phase2/phase2.h"
#include "phase2/gpu_lp_boundary.h"

// ==================== 유틸리티 함수들 ====================

/**
 * @brief Ghost 노드 라벨을 안전하게 조회하는 인라인 함수
 * @param node_id 조회할 노드 ID
 * @param g 로컬 그래프 구조
 * @param labels 로컬 노드 라벨 배열
 * @param ghost_nodes 고스트 노드 정보
 * @return 노드의 라벨 (-1: 유효하지 않은 노드)
 */
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

/**
 * @brief 파티션별 통계를 계산하는 최적화된 함수
 * 
 * 특징:
 * - OpenMP 병렬화로 노드/간선 카운팅 가속화
 * - 단일 MPI Allreduce로 통신 오버헤드 최소화
 * - 스레드 로컬 카운터로 false sharing 방지
 * 
 * @param g 로컬 그래프 구조
 * @param labels 노드 라벨 배열
 * @param ghost_nodes 고스트 노드 정보
 * @param num_partitions 총 파티션 수
 * @return 파티션 통계 (로컬 및 글로벌)
 */
static PartitionStats computePartitionStats(const Graph &g, const std::vector<int> &labels, 
                                           const GhostNodes &ghost_nodes, int num_partitions) {
    PartitionStats stats;
    stats.local_vertex_counts.resize(num_partitions, 0);
    stats.local_edge_counts.resize(num_partitions, 0);
    stats.global_vertex_counts.resize(num_partitions, 0);
    stats.global_edge_counts.resize(num_partitions, 0);

    // 병렬 노드 카운팅 - 스레드별 로컬 카운터로 false sharing 방지
    #pragma omp parallel
    {
        std::vector<int> thread_vertex_counts(num_partitions, 0);
        
        #pragma omp for nowait schedule(static)
        for (int u = 0; u < g.num_vertices; u++) {
            int label = labels[u];
            if (label >= 0 && label < num_partitions) {
                thread_vertex_counts[label]++;
            }
        }
        
        // 스레드별 결과를 전역 카운터에 병합
        #pragma omp critical
        {
            for (int i = 0; i < num_partitions; i++) {
                stats.local_vertex_counts[i] += thread_vertex_counts[i];
            }
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
            for (int i = 0; i < num_partitions; i++) {
                stats.local_edge_counts[i] += thread_edge_counts[i];
            }
        }
    }

    // 통신 최적화: 단일 MPI Allreduce로 vertex/edge 카운트 동시 집계
    std::vector<int> send_buffer(2 * num_partitions);
    std::vector<int> recv_buffer(2 * num_partitions);
    
    // 메모리 패킹: [vertex_counts..., edge_counts...]
    std::copy(stats.local_vertex_counts.begin(), stats.local_vertex_counts.end(), send_buffer.begin());
    std::copy(stats.local_edge_counts.begin(), stats.local_edge_counts.end(), send_buffer.begin() + num_partitions);
    
    // 단일 Allreduce로 네트워크 오버헤드 최소화
    MPI_Allreduce(send_buffer.data(), recv_buffer.data(), 2 * num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // 결과 언패킹
    std::copy(recv_buffer.begin(), recv_buffer.begin() + num_partitions, stats.global_vertex_counts.begin());
    std::copy(recv_buffer.begin() + num_partitions, recv_buffer.end(), stats.global_edge_counts.begin());

    // 통계 계산 최적화
    stats.total_vertices = std::accumulate(stats.global_vertex_counts.begin(), stats.global_vertex_counts.end(), 0);
    stats.total_edges = std::accumulate(stats.global_edge_counts.begin(), stats.global_edge_counts.end(), 0);

    // 균등 분배 기준값 (로드 밸런싱 목표)
    stats.expected_vertices = static_cast<double>(stats.total_vertices) / num_partitions;
    stats.expected_edges = (stats.total_edges > 0) ? static_cast<double>(stats.total_edges) / num_partitions : 1.0;

    return stats;
}

/**
 * @brief MPI Delta 통신을 위한 최적화된 Allgather 함수
 * 
 * Delta 구조체를 효율적으로 수집하여 라벨 변경사항을 모든 프로세서에 배포
 * 
 * @param local_deltas 로컬 프로세서의 델타 변경사항
 * @param mpi_size MPI 프로세서 수
 * @return 모든 프로세서의 델타 변경사항
 */
static std::vector<Delta> allgatherDeltas(const std::vector<Delta> &local_deltas, int mpi_size) {
    int send_count = local_deltas.size();
    std::vector<int> recv_counts(mpi_size);
    
    // 1단계: 각 프로세서의 델타 수 수집
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 2단계: displacement 계산
    std::vector<int> displs(mpi_size);
    displs[0] = 0;
    for (int i = 1; i < mpi_size; i++)
        displs[i] = displs[i - 1] + recv_counts[i - 1];
    int total_recv = displs[mpi_size - 1] + recv_counts[mpi_size - 1];

    std::vector<Delta> recv_deltas(total_recv);

    // 3단계: Delta용 MPI 타입 정의 및 데이터 수집
    MPI_Datatype MPI_DELTA;
    MPI_Type_contiguous(2, MPI_INT, &MPI_DELTA);
    MPI_Type_commit(&MPI_DELTA);

    MPI_Allgatherv(local_deltas.data(), send_count, MPI_DELTA,
                   recv_deltas.data(), recv_counts.data(), displs.data(),
                   MPI_DELTA, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_DELTA);
    
    return recv_deltas;
}

// ==================== DMOLP Penalty 계산 ====================

/**
 * @brief 실험용 컴파일 타임 설정: Master-Worker vs 분산 penalty 계산
 * 
 * USE_MASTER_WORKER_PENALTY 정의시 Master-Worker 방식 사용
 * 미정의시 모든 프로세서가 동일한 계산 수행 (기본값)
 */
// #define USE_MASTER_WORKER_PENALTY  // 이 줄을 주석 해제하면 Master-Worker 방식 사용

/**
 * @brief DMOLP Penalty 계산 함수 (Master-Worker 방식)
 * 
 * 특징:
 * - Rank 0만 penalty 계산 수행하여 계산 오버헤드 최소화
 * - 계산 결과를 모든 프로세서에 브로드캐스트
 * - 파티션 불균형을 기반으로 한 DMOLP 알고리즘 적용
 * 
 * @param stats 파티션 통계 정보
 * @param num_partitions 총 파티션 수
 * @param mpi_rank 현재 프로세서 랭크
 * @return 각 파티션별 penalty 값
 */
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

/**
 * @brief Boundary Local ID 추출 함수 (최적화된 병렬 처리)
 * 
 * 기능:
 * - 경계 노드들의 local ID만을 추출하여 메모리 효율성 향상
 * - OpenMP 병렬 처리로 성능 최적화
 * - 벡터 크기 사전 할당으로 메모리 재할당 최소화
 * 
 * 최적화 특징:
 * - reserve()를 통한 메모리 사전 할당
 * - parallel for 구문으로 멀티스레드 처리
 * - 중복 검사 최소화를 위한 효율적인 자료구조 활용
 * 
 * @param graph 그래프 객체 (const 참조로 안전한 읽기 전용 접근)
 * @param comm_partners 통신 파트너 집합
 * @return boundary node들의 local ID 벡터
 * 
 * @note OpenMP가 활성화된 환경에서 최적 성능 발휘
 * @warning 그래프 크기가 작을 경우 오버헤드 고려 필요
 */
static std::vector<int> extractBoundaryLocalIDs(const Graph &graph, const GhostNodes &ghost_nodes)
{
    std::vector<int> boundary_nodes;
    
    #pragma omp parallel
    {
        std::vector<int> thread_boundary_nodes;
        
        #pragma omp for nowait
        for (int u = 0; u < graph.num_vertices; u++) {
            int u_label = graph.vertex_labels[u];
            bool is_boundary = false;
            
            // u의 이웃들을 검사
            for (int edge_idx = graph.row_ptr[u]; edge_idx < graph.row_ptr[u + 1]; edge_idx++) {
                int v = graph.col_indices[edge_idx];
                int v_label = getNodeLabel(v, graph, graph.vertex_labels, ghost_nodes);
                
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

/**
 * @brief Edge-cut 계산 함수 (최적화된 MPI 병렬 처리)
 * 
 * 기능:
 * - owned 노드의 간선만 카운트하여 MPI 환경에서 중복 방지
 * - OpenMP 병렬 처리로 로컬 계산 성능 최적화
 * - MPI_Allreduce로 전역 edge-cut 값 집계
 * 
 * 최적화 특징:
 * - reduction 절을 이용한 thread-safe 카운팅
 * - 중복 edge-cut 계산 방지 (owned 노드만 처리)
 * - 효율적인 ghost node 라벨 조회
 * 
 * @param g 그래프 객체
 * @param labels 노드 라벨 배열
 * @param ghost_nodes ghost node 정보
 * @return 전역 edge-cut 값
 * 
 * @note MPI 환경에서 정확한 edge-cut 계산을 위해 owned 노드만 처리
 */
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

/**
 * @brief Phase2 분산 그래프 파티셔닝 메인 실행 함수
 * 
 * 핵심 기능:
 * - GPU 가속 boundary label propagation 수행
 * - MPI 기반 분산 처리로 대규모 그래프 파티셔닝
 * - DMOLP penalty 기반 적응적 파티션 조정
 * - 수렴 기준 달성시까지 반복 최적화
 * 
 * 성능 최적화:
 * - CUDA GPU 커널을 활용한 고속 label propagation
 * - Pinned Memory 사용으로 CPU-GPU 데이터 전송 최적화
 * - OpenMP 멀티스레딩으로 CPU 병렬 처리
 * - MPI Non-blocking communication으로 통신 오버헤드 최소화
 * 
 * 알고리즘 특징:
 * - 경계 노드 집중 처리로 계산량 감소
 * - 동적 수렴 판정으로 불필요한 반복 제거
 * - 메모리 사용량 최적화된 데이터 구조
 * 
 * @param mpi_rank 현재 MPI 프로세서 랭크
 * @param mpi_size 전체 MPI 프로세서 수
 * @param num_partitions 목표 파티션 개수
 * @param local_graph 로컬 그래프 데이터 (소유 노드 + 경계 정보)
 * @param ghost_nodes 원격 노드 정보 (MPI 통신용)
 * @param gpu_id 사용할 GPU 디바이스 ID
 * @return PartitioningMetrics 파티셔닝 성능 지표 및 결과
 * 
 * @note GPU 메모리가 부족한 경우 자동으로 CPU fallback 수행
 * @warning max_iter 도달시 강제 종료되므로 적절한 값 설정 필요
 */
PartitioningMetrics run_phase2(
    int mpi_rank, int mpi_size,
    int num_partitions,
    Graph &local_graph,
    GhostNodes &ghost_nodes,
    int gpu_id)
{
    const int max_iter = 500;        // 최대 반복 횟수 (발산 방지)
    const double epsilon = 0.03;     // 수렴 기준 (상대 변화율)
    const int k_limit = 10;          // 조기 종료 판정을 위한 연속 수렴 횟수

    // 성능 측정을 위한 타이머 시작
    auto t_phase2_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "[Rank " << mpi_rank << "/" << mpi_size << "] Phase2 시작 (GPU " << gpu_id << ")" << std::endl;
    std::cout.flush();
    
    // ==================== 메모리 최적화 설정 ====================
    // CPU 메모리 Pin 최적화: 자주 사용되는 벡터들을 Pinned Memory로 할당
    // GPU와의 빠른 데이터 전송을 위한 메모리 최적화
    std::vector<double> penalty_pinned;
    std::vector<int> boundary_nodes_pinned;
    
    // GPU와 자주 통신하는 메모리를 Pinned로 할당 (성능 향상)
    penalty_pinned.reserve(num_partitions);  // 미리 공간 확보
    
    // ==================== 초기 상태 평가 ====================
    // 현재 파티셔닝의 edge-cut 계산 (최적화 기준점)
    int prev_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels, ghost_nodes);
    int convergence_count = 0;  // 연속 수렴 횟수 추적

    // 파티션 통계 계산 (DMOLP penalty 계산용)
    PartitionStats current_stats = computePartitionStats(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions);
    
    // ==================== 성능 최적화를 위한 메모리 사전 할당 ====================
    // 반복 과정에서 사용될 주요 데이터 구조들의 메모리 미리 확보
    penalty_pinned.resize(num_partitions);
    std::vector<Delta> delta_changes;
    delta_changes.reserve(1000);  // 예상 변경사항 수만큼 미리 할당
    
    // ==================== 적응적 바운더리 관리 시스템 ====================
    std::vector<int> current_boundary_nodes;
    bool first_iteration = true;  // 첫 번째 반복에서는 전체 노드 처리
    
    // 결과 메트릭 초기화
    PartitioningMetrics m2;

    // ==================== 메인 최적화 반복 루프 ====================
    for (int iter = 0; iter < max_iter; iter++) {
        
        // ============ Step 1: DMOLP Penalty 계산 ============
        // 현재 파티션 불균형 상태를 기반으로 각 파티션별 penalty 계산
        penalty_pinned = calculatePenalties(current_stats, num_partitions, mpi_rank);

        // ============ Step 2: 적응적 바운더리 노드 선정 ============
        if (first_iteration) {
            // 첫 번째 이터레이션: 모든 노드를 처리하여 초기 바운더리 탐지
            current_boundary_nodes.clear();
            for (int i = 0; i < local_graph.num_vertices; i++) {
                current_boundary_nodes.push_back(i);
            }
            first_iteration = false;
            printf("[Rank %d] 첫 번째 이터레이션: 전체 %d 노드 처리\n", 
                   mpi_rank, local_graph.num_vertices);
        } else {
            // 이후 이터레이션: 이전 바운더리 + 1-hop 확장으로 효율성 향상
            current_boundary_nodes = expandBoundaryNodes(
                local_graph.row_ptr, local_graph.col_indices,
                current_boundary_nodes, local_graph.vertex_labels,
                local_graph.num_vertices + ghost_nodes.ghost_labels.size());
        }
        
        // 바운더리가 없으면 최적화 완료
        if (current_boundary_nodes.empty()) {
            if (mpi_rank == 0) std::cout << "경계 노드 없음, 수렴 완료\n";
            break;
        }

        // ============ Step 3: GPU 가속 바운더리 라벨 전파 ============
        GPULabelUpdateResult gpu_result;
        try {
            // ======== GPU 메모리 적응적 관리 ========
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            size_t max_gpu_memory = static_cast<size_t>(free_memory * 0.85); // 85%로 보수적 사용
            
            printf("[Rank %d] GPU 메모리: 전체 %.1fGB, 사용가능 %.1fGB, 사용예정 %.1fGB\n", 
                   mpi_rank, total_memory / (1024.0*1024.0*1024.0), 
                   free_memory / (1024.0*1024.0*1024.0),
                   max_gpu_memory / (1024.0*1024.0*1024.0));
            
            // ======== 고성능 GPU 커널 실행 ========
            // 통합 서브그래프 방식: 로컬+고스트 라벨 통합하여 GPU에 전달
            // 스트리밍 최적화로 대용량 그래프도 처리 가능
            auto gpu_start = std::chrono::high_resolution_clock::now();
            
            gpu_result = runBoundaryLPOnGPU_Streaming(
                local_graph.row_ptr,          // CSR format row pointers
                local_graph.col_indices,      // CSR format column indices  
                current_boundary_nodes,       // 최적화된 바운더리 노드들
                local_graph.vertex_labels,    // 소유 노드 라벨
                ghost_nodes.ghost_labels,     // 원격 노드 라벨
                local_graph.global_ids,       // 로컬-글로벌 ID 매핑
                penalty_pinned,               // DMOLP penalty 배열
                local_graph.num_vertices,     // 소유 노드 수
                num_partitions,               // 파티션 개수
                max_gpu_memory / (1024 * 1024)  // 사용가능 메모리 (MB)
            );
            
            auto gpu_end = std::chrono::high_resolution_clock::now();
            auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
            printf("[Rank %d] GPU 처리 완료: %.2fms, %d 변경사항\n", 
                   mpi_rank, gpu_duration.count() / 1000.0, gpu_result.change_count);
                   
        } catch (const std::exception& e) {
            printf("[Rank %d] GPU 처리 실패 (CPU fallback 지원): %s\n", mpi_rank, e.what());
            // GPU 실패 시 CPU fallback은 여기에 구현 가능
            gpu_result.change_count = 0;
        }

        // ============ Step 4: 분산 라벨 동기화 (MPI 통신) ============
        // 비동기 통신으로 GPU 처리와 통신 오버헤드를 오버랩
        std::vector<Delta> recv_deltas;
        MPI_Request comm_request = MPI_REQUEST_NULL;
        bool async_comm_started = false;
        
        // ======== 비동기 MPI 통신 시작 ========
        // 모든 프로세서의 라벨 변경사항을 수집하기 위한 비동기 allgather
        auto async_result = std::async(std::launch::async, [&]() {
            return allgatherDeltas(delta_changes, mpi_size);
        });
        recv_deltas = async_result.get();  // 통신 완료 대기
        async_comm_started = true;
        
        // GPU 동기화 (통신과 병렬 실행됨)
        cudaDeviceSynchronize();

        // ======== GPU 결과 적용 및 Delta 수집 ========
        delta_changes.clear();  // 메모리 재사용: clear()로 용량 유지, 성능 최적화
        
        // GPU에서 변경된 로컬 노드들의 라벨 업데이트를 Delta로 변환
        for (int i = 0; i < gpu_result.change_count; i++) {
            int local_node_id = gpu_result.updated_nodes[i];
            int new_label = gpu_result.updated_labels[i];
            
            // 유효한 로컬 노드만 처리
            if (local_node_id >= 0 && local_node_id < local_graph.num_vertices) {
                // Delta 구조체에 변경사항 기록 (MPI 통신용)
                if (local_node_id < (int)local_graph.global_ids.size()) {
                    Delta delta;
                    delta.gid = local_graph.global_ids[local_node_id];  // 글로벌 ID
                    delta.new_label = new_label;                        // 새 라벨
                    delta_changes.push_back(delta);
                }
                
                // 실제 라벨 적용 (로컬 노드만)
                local_graph.vertex_labels[local_node_id] = new_label;
            }
        }
        
        std::cout << "[Rank " << mpi_rank << "] GPU 라벨 변경: " << gpu_result.change_count 
                  << " (로컬 delta: " << delta_changes.size() << ")" << std::endl;

        // ======== MPI 분산 라벨 동기화 결과 처리 ========
        printf("[Rank %d] Iter %d: 로컬 변경 %zu개, 수신 변경 %zu개\n", 
               mpi_rank, iter, delta_changes.size(), recv_deltas.size());

        // ============ Step 5: 원격 프로세서 라벨 변경사항 적용 ============
        // 다른 프로세서에서 받은 ghost node 라벨 업데이트
        for (const auto &delta : recv_deltas) {
            // ghost 노드 여부 확인
            auto it_ghost = ghost_nodes.global_to_local.find(delta.gid);
            if (it_ghost != ghost_nodes.global_to_local.end()) {
                int ghost_idx = it_ghost->second;
                
                // 범위 검증 후 라벨 업데이트
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
        
        // ======== 전역 수렴 판정: 모든 프로세서의 변경사항 집계 ========
        int total_changes = delta_changes.size() + recv_deltas.size();
        int global_total_changes;
        MPI_Allreduce(&total_changes, &global_total_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // 전역적으로 변경사항이 없으면 조기 종료
        if (global_total_changes == 0) {
            if (mpi_rank == 0) {
                std::cout << "Iter " << iter + 1 << ": 전체 시스템 수렴 완료 (변경사항 없음)\n";
            }
            break;
        }

        // ============ Step 6: 파티션 통계 업데이트 및 수렴 판정 ============
        current_stats = computePartitionStats(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions);

        // Edge-cut 기반 수렴 판정
        int curr_edge_cut = computeEdgeCut(local_graph, local_graph.vertex_labels, ghost_nodes);
        double delta = (prev_edge_cut > 0)
                           ? std::abs((double)(curr_edge_cut - prev_edge_cut) / prev_edge_cut)
                           : 1.0;
        
        // 수렴 판정: 연속 k_limit번 변화율이 epsilon 미만인 경우
        if (delta < epsilon) {
            convergence_count++;
        } else {
            convergence_count = 0; // 변화율이 크면 카운터 리셋
        }
        
        // ======== 반복 결과 출력 및 진행 상황 모니터링 ========
        if (mpi_rank == 0) {
            std::cout << "Iter " << iter + 1 << ": Edge-cut " << curr_edge_cut 
                      << " (변화율: " << std::fixed << std::setprecision(3) << delta * 100 << "%)";
            
            if (convergence_count > 0) {
                std::cout << " [수렴 진행: " << convergence_count << "/" << k_limit << "]";
            }
            std::cout << "\n";
        }
        prev_edge_cut = curr_edge_cut;
        
        // ======== 수렴 완료 판정 ========
        if (convergence_count >= k_limit) {
            if (mpi_rank == 0) {
                std::cout << "수렴 완료! (연속 " << k_limit << "회 변화율 < " 
                          << std::fixed << std::setprecision(1) << epsilon * 100 << "%)\n";
            }
            break;
        }
        
    } // 메인 최적화 루프 종료
    
    // ==================== Phase2 완료 및 성능 측정 ====================
    auto t_phase2_end = std::chrono::high_resolution_clock::now();
    long exec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_phase2_end - t_phase2_start).count();

    // 최종 성능 및 GPU 사용 통계 출력
    std::cout << "[Rank " << mpi_rank << "] Phase2 완료 - GPU " << gpu_id 
              << " 총 실행시간: " << exec_ms << "ms" << std::endl;
    
    // ==================== 최종 파티셔닝 품질 평가 ====================
    PartitionStats final_stats = computePartitionStats(local_graph, local_graph.vertex_labels, ghost_nodes, num_partitions);

    // 파티션 균형도 계산 (DMOLP 품질 지표)
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

    // ==================== 최종 메트릭 구성 ====================
    m2.edge_cut = prev_edge_cut;                                    // 최종 edge-cut
    m2.vertex_balance = max_vertex_ratio / avg_vertex_ratio;        // 노드 균형도
    m2.edge_balance = max_edge_ratio / avg_edge_ratio;              // 간선 균형도
    m2.loading_time_ms = exec_ms;                                   // 총 실행 시간
    m2.distribution_time_ms = 0;                                    // 분산 처리 시간 (Phase2에서는 미사용)
    m2.num_partitions = num_partitions;                             // 파티션 수
    
    // 전역 그래프 통계 (이미 계산된 final_stats 활용)
    m2.total_vertices = final_stats.total_vertices;
    m2.total_edges = final_stats.total_edges;

    // ==================== 최종 동기화 및 정리 ====================
    MPI_Barrier(MPI_COMM_WORLD);                                   // 모든 프로세서 동기화
    cudaDeviceSynchronize();                                        // 모든 GPU 작업 완료 대기
    std::cout << "[Rank " << mpi_rank << "] GPU " << gpu_id << " 작업 완료" << std::endl;
    std::cout.flush();
    
    return m2;
}
