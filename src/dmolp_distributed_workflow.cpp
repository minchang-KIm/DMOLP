
// --- 함수 구현부 (static) ---
#include <vector>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mpi.h>
#include <omp.h>
#include "types.h"


static void calculateRatios(int mpi_rank, int num_partitions, const Graph& local_graph, std::vector<int>& vertex_labels, std::vector<PartitionInfo>& PI) {
    std::cout << "Step1 Rank " << mpi_rank << ": RV, RE 계산\n";
    std::vector<int> vertex_counts(num_partitions, 0);
    std::vector<int> edge_counts(num_partitions, 0);
    int n_v = std::min(local_graph.num_vertices, (int)vertex_labels.size());
#pragma omp parallel for
    for (int i = 0; i < n_v; ++i) {
        int label = vertex_labels[i];
        if (label >= 0 && label < num_partitions) {
#pragma omp atomic
            vertex_counts[label]++;
        }
    }
    int n_e = std::min(local_graph.num_vertices, (int)vertex_labels.size());
#pragma omp parallel for
    for (int u = 0; u < n_e; ++u) {
        int u_label = vertex_labels[u];
        if (u_label < 0 || u_label >= num_partitions) continue;
        if (u < (int)local_graph.row_ptr.size() - 1) {
            for (int edge_idx = local_graph.row_ptr[u]; edge_idx < local_graph.row_ptr[u + 1] && edge_idx < (int)local_graph.col_indices.size(); ++edge_idx) {
                int v = local_graph.col_indices[edge_idx];
                if (v >= 0 && v < (int)vertex_labels.size()) {
                    int v_label = vertex_labels[v];
                    if (v_label == u_label) {
#pragma omp atomic
                        edge_counts[u_label]++;
                    }
                }
            }
        }
    }
    std::vector<int> global_vertex_counts(num_partitions, 0);
    std::vector<int> global_edge_counts(num_partitions, 0);
    MPI_Allreduce(vertex_counts.data(), global_vertex_counts.data(), num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(edge_counts.data(), global_edge_counts.data(), num_partitions, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int total_vertices = 0, total_edges = 0;
    for (int i = 0; i < num_partitions; ++i) {
        total_vertices += global_vertex_counts[i];
        total_edges += global_edge_counts[i];
    }
    for (int i = 0; i < num_partitions; ++i) {
        PI[i].RV = (total_vertices > 0) ? (double)global_vertex_counts[i] / ((double)total_vertices / num_partitions) : 1.0;
        PI[i].RE = (total_edges > 0) ? (double)global_edge_counts[i] / ((double)total_edges / num_partitions) : 1.0;
    }
    std::cout << "RV/RE계산완료 Rank " << mpi_rank << ": 총 " << total_vertices << "개 정점, " << total_edges << "개 간선\n";
}

// Step 2: 전체 imbalance 계산
static void calculateImbalance(int mpi_rank, int num_partitions, std::vector<PartitionInfo>& PI) {
    std::cout << "Step2 Rank " << mpi_rank << ": 불균형 계산\n";
    double rv_mean = 1.0, re_mean = 1.0;
    double rv_variance = 0.0, re_variance = 0.0;
    for (int i = 0; i < num_partitions; ++i) {
        rv_variance += (PI[i].RV - rv_mean) * (PI[i].RV - rv_mean);
        re_variance += (PI[i].RE - re_mean) * (PI[i].RE - re_mean);
    }
    rv_variance /= num_partitions;
    re_variance /= num_partitions;
    double total_variance = rv_variance + re_variance;
    double imb_rv = 0.0, imb_re = 0.0;
    if (total_variance > 0) {
        imb_rv = rv_variance / total_variance;
        imb_re = re_variance / total_variance;
        for (int i = 0; i < num_partitions; ++i) {
            PI[i].imb_RV = imb_rv;
            PI[i].imb_RE = imb_re;
            PI[i].G_RV = (1.0 - PI[i].RV) / num_partitions;
            PI[i].G_RE = (1.0 - PI[i].RE) / num_partitions;
            PI[i].P_L = imb_rv * PI[i].G_RV + imb_re * PI[i].G_RE;
        }
    } else {
        for (int i = 0; i < num_partitions; ++i) {
            PI[i].imb_RV = 0.0;
            PI[i].imb_RE = 0.0;
            PI[i].G_RV = 0.0;
            PI[i].G_RE = 0.0;
            PI[i].P_L = 0.0;
        }
    }
    std::cout << "불균형계산완료 Rank " << mpi_rank << ": RV분산=" << rv_variance << ", RE분산=" << re_variance << "\n";
}

// Step 3: Edge-cut 계산 및 BV/NV 추출
static void calculateEdgeCutAndExtractBoundary(int mpi_rank, int mpi_size, const Graph& local_graph, std::vector<int>& vertex_labels, std::vector<PartitionInfo>& PI, int& current_edge_cut, int& previous_edge_cut, double& edge_rate, std::vector<int>& BV, std::unordered_map<int, int>& NV, PartitionUpdate& PU) {
    std::cout << "Step3 Rank " << mpi_rank << ": Edge-cut 계산 및 BV/NV 추출\n";
    if (local_graph.num_vertices == 0) {
        previous_edge_cut = current_edge_cut;
        current_edge_cut = 0;
        edge_rate = 0.0;
        BV.clear();
        NV.clear();
        return;
    }
    previous_edge_cut = current_edge_cut;
    int local_edge_cut = 0;
    for (int u = 0; u < local_graph.num_vertices; ++u) {
        int u_label = vertex_labels[u];
        for (int edge_idx = local_graph.row_ptr[u]; edge_idx < local_graph.row_ptr[u + 1]; ++edge_idx) {
            int v = local_graph.col_indices[edge_idx];
            if (v < local_graph.num_vertices) {
                int v_label = vertex_labels[v];
                if (u_label != v_label) {
                    local_edge_cut++;
                }
            }
        }
    }
    MPI_Allreduce(&local_edge_cut, &current_edge_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (previous_edge_cut > 0) {
        edge_rate = static_cast<double>(previous_edge_cut - current_edge_cut) / previous_edge_cut;
    } else {
        edge_rate = 0.0;
    }
    // BV/NV 추출 (OpenMP)
    BV.clear();
    NV.clear();
    std::vector<std::vector<int>> thread_boundary_vertices(omp_get_max_threads());
    std::vector<std::unordered_map<int, int>> thread_neighbor_vertices(omp_get_max_threads());
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
#pragma omp for schedule(dynamic, 1000)
        for (int u = 0; u < local_graph.num_vertices; ++u) {
            int u_label = vertex_labels[u];
            bool is_boundary = false;
            for (int edge_idx = local_graph.row_ptr[u]; edge_idx < local_graph.row_ptr[u + 1]; ++edge_idx) {
                int v = local_graph.col_indices[edge_idx];
                if (v < local_graph.num_vertices) {
                    int v_label = vertex_labels[v];
                    if (u_label != v_label) {
                        is_boundary = true;
                        thread_neighbor_vertices[thread_id][v] = v_label;
                    }
                }
            }
            if (is_boundary) {
                thread_boundary_vertices[thread_id].push_back(u);
            }
        }
    }
    for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
        BV.insert(BV.end(), thread_boundary_vertices[tid].begin(), thread_boundary_vertices[tid].end());
        for (auto& pair : thread_neighbor_vertices[tid]) {
            NV[pair.first] = pair.second;
        }
    }
}

// Step 4: Dynamic Label Propagation
static void performDynamicLabelPropagation(int mpi_rank, const Graph& local_graph, std::vector<int>& vertex_labels, std::vector<PartitionInfo>& PI, std::vector<int>& BV, PartitionUpdate& PU, const std::vector<int>& remote_labels, int local_offset) {
    std::cout << "Step4 Rank " << mpi_rank << ": Dynamic LP 수행 (BV.size=" << BV.size() << ")\n";
    PU.PU_RO.clear(); PU.PU_OV.clear(); PU.PU_ON.clear();
    int updates_count = 0;
#pragma omp parallel for reduction(+:updates_count)
    for (size_t idx = 0; idx < BV.size(); ++idx) {
        int u = BV[idx];
        int old_label = vertex_labels[u];
        std::unordered_map<int, int> label_count;
        for (int edge_idx = local_graph.row_ptr[u]; edge_idx < local_graph.row_ptr[u + 1]; ++edge_idx) {
            int v = local_graph.col_indices[edge_idx];
            int v_label;
            if (v >= 0) {
                v_label = vertex_labels[v]; // 로컬 노드
            } else {
                int global_v = -1 - v;
                v_label = remote_labels[global_v]; // remote 노드
            }
            label_count[v_label]++;
        }
        int best_label = old_label;
        double best_score = -1e9;
        for (auto& kv : label_count) {
            int L = kv.first;
            int cnt = kv.second;
            double score = cnt * (1.0 + PI[L].P_L);
            if (score > best_score) {
                best_score = score;
                best_label = L;
            }
        }
        if (best_label != old_label) {
            vertex_labels[u] = best_label;
            updates_count++;
            for (int edge_idx = local_graph.row_ptr[u]; edge_idx < local_graph.row_ptr[u + 1]; ++edge_idx) {
                int v = local_graph.col_indices[edge_idx];
                if (vertex_labels[v] == old_label && std::find(BV.begin(), BV.end(), v) == BV.end()) {
#pragma omp critical
                    PU.PU_RO.push_back(v);
                }
            }
#pragma omp critical
            PU.PU_OV.push_back(u);
            for (int edge_idx = local_graph.row_ptr[u]; edge_idx < local_graph.row_ptr[u + 1]; ++edge_idx) {
                int v = local_graph.col_indices[edge_idx];
#pragma omp critical
                PU.PU_ON.push_back({u, v});
            }
        }
    }
    int total_updates = 0;
    MPI_Allreduce(&updates_count, &total_updates, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "Step4 완료 Rank " << mpi_rank << ": 라벨 변경 " << total_updates << "개\n";
}

// Step 5: 파티션 업데이트 교환
static void exchangePartitionUpdates(int mpi_rank, int mpi_size, std::vector<int>& BV, std::unordered_map<int, int>& NV, PartitionUpdate& PU) {
    std::cout << "Step5 Rank " << mpi_rank << ": 파티션 업데이트 교환 (PU_OV/PU_ON)\n";
    int ov_size = PU.PU_OV.size();
    int on_size = PU.PU_ON.size();
    std::vector<int> all_ov_sizes(mpi_size);
    std::vector<int> all_on_sizes(mpi_size);
    MPI_Allgather(&ov_size, 1, MPI_INT, all_ov_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&on_size, 1, MPI_INT, all_on_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int total_ov = 0, total_on = 0;
    for (int s : all_ov_sizes) total_ov += s;
    for (int s : all_on_sizes) total_on += s;
    std::vector<int> PU_RV(total_ov);
    std::vector<std::pair<int, int>> PU_RN(total_on);
    std::vector<int> ov_displs(mpi_size, 0), on_displs(mpi_size, 0);
    for (int i = 1; i < mpi_size; ++i) {
        ov_displs[i] = ov_displs[i-1] + all_ov_sizes[i-1];
        on_displs[i] = on_displs[i-1] + all_on_sizes[i-1];
    }
    MPI_Allgatherv(PU.PU_OV.data(), ov_size, MPI_INT, PU_RV.data(), all_ov_sizes.data(), ov_displs.data(), MPI_INT, MPI_COMM_WORLD);
    // PU_ON은 std::pair<int,int>이므로, int*로 변환 필요
    std::vector<int> PU_ON_flat(PU.PU_ON.size() * 2);
    for (size_t i = 0; i < PU.PU_ON.size(); ++i) {
        PU_ON_flat[2*i] = PU.PU_ON[i].first;
        PU_ON_flat[2*i+1] = PU.PU_ON[i].second;
    }
    std::vector<int> PU_RN_flat(total_on * 2);
    MPI_Allgatherv(PU_ON_flat.data(), on_size * 2, MPI_INT, PU_RN_flat.data(), all_on_sizes.data(), on_displs.data(), MPI_INT, MPI_COMM_WORLD);
    // PU_RO는 로컬에서 BV/NV로 반영
    for (int v : PU.PU_RO) {
        if (std::find(BV.begin(), BV.end(), v) == BV.end()) {
            BV.push_back(v);
            NV[v] = -1; // 라벨 정보 없음
        }
    }
    std::cout << "Step5 완료 Rank " << mpi_rank << ": PU_RV=" << PU_RV.size() << ", PU_RN=" << PU_RN_flat.size()/2 << ", BV=" << BV.size() << "\n";
}

// Step 6: 수렴 확인 (Edge-cut 변화량, 반복 카운트)
static bool checkConvergence(int mpi_rank, int& convergence_count, int current_edge_cut, int previous_edge_cut) {
    double local_delta = (previous_edge_cut > 0) ? std::abs((double)(current_edge_cut - previous_edge_cut) / previous_edge_cut) : 1.0;
    double global_delta = 0.0;
    MPI_Allreduce(&local_delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (global_delta < 0.03) {
        convergence_count++;
    } else {
        convergence_count = 0;
    }
    if (convergence_count >= 10) {
        if (mpi_rank == 0) std::cout << "수렴 조건 달성! (delta=" << global_delta << ", count=" << convergence_count << ")\n";
        return true;
    }
    return false;
}

// Step 7: 최종 결과 출력 (간단 버전)
static void printFinalResults(int mpi_rank, int num_partitions, const std::vector<PartitionInfo>& PI, int current_edge_cut, long execution_time_ms, const Phase1Metrics& phase1_metrics) {
    if (mpi_rank != 0) return;
    std::cout << "\n=== 최종 결과 (7단계 알고리즘) ===\n";
    std::cout << "Edge-cut: " << current_edge_cut << "\n";
    double max_vertex_ratio = 0.0, max_edge_ratio = 0.0;
    for (int i = 0; i < num_partitions; ++i) {
        max_vertex_ratio = std::max(max_vertex_ratio, PI[i].RV);
        max_edge_ratio = std::max(max_edge_ratio, PI[i].RE);
    }
    std::cout << "Vertex Balance (VB): " << max_vertex_ratio << "\n";
    std::cout << "Edge Balance (EB): " << max_edge_ratio << "\n";
    std::cout << "Execution Time: " << execution_time_ms << " ms\n";
}



// --- 함수 선언부 (static) ---
// (구조체 정의는 types.h에서 제공)
static void calculateRatios(int mpi_rank, int num_partitions, const Graph& local_graph, std::vector<int>& vertex_labels, std::vector<PartitionInfo>& PI);
static void calculateImbalance(int mpi_rank, int num_partitions, std::vector<PartitionInfo>& PI);
static void calculateEdgeCutAndExtractBoundary(
    int mpi_rank, int mpi_size, const Graph& local_graph, std::vector<int>& vertex_labels, std::vector<PartitionInfo>& PI,
    int& current_edge_cut, int& previous_edge_cut, double& edge_rate, std::vector<int>& BV, std::unordered_map<int, int>& NV, PartitionUpdate& PU);
static void performDynamicLabelPropagation(int mpi_rank, const Graph& local_graph, std::vector<int>& vertex_labels, std::vector<PartitionInfo>& PI, std::vector<int>& BV, PartitionUpdate& PU);
static void exchangePartitionUpdates(int mpi_rank, int mpi_size, std::vector<int>& BV, std::unordered_map<int, int>& NV, PartitionUpdate& PU);
static bool checkConvergence(int mpi_rank, int& convergence_count, int current_edge_cut, int previous_edge_cut);
static void printFinalResults(int mpi_rank, int num_partitions, const std::vector<PartitionInfo>& PI, int current_edge_cut, long execution_time_ms, const Phase1Metrics& phase1_metrics);

// --- 메인 함수형 워크플로우 ---
void dmolp_distributed_workflow_run(int argc, char** argv, const Graph& local_graph, const std::vector<int>& vertex_labels_in, const Phase1Metrics& phase1_metrics) {
    int mpi_rank = 0, mpi_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_partitions = phase1_metrics.partition_vertex_counts.size();
    std::cout << "[DMOLP] run() called (rank=" << mpi_rank << ")\n";
    std::vector<int> vertex_labels = vertex_labels_in;
    // remote_labels: 전체 그래프의 라벨(모든 rank에서 Allgather)
    int global_num_vertices = 0;
    if (!phase1_metrics.partition_vertex_counts.empty()) {
        for (int cnt : phase1_metrics.partition_vertex_counts) global_num_vertices += cnt;
    }
    std::vector<int> remote_labels(global_num_vertices, -1);
    std::vector<PartitionInfo> PI(num_partitions);
    PartitionUpdate PU;
    std::vector<int> BV;
    std::unordered_map<int, int> NV;
    int current_edge_cut = 0, previous_edge_cut = 0;
    double edge_rate = 0.0;
    int convergence_count = 0;

    int max_iters = 100;
    int iter = 0;
    bool converged = false;
    double t_start = MPI_Wtime();
    while (iter < max_iters && !converged) {
        // Step 0: 모든 rank의 vertex_labels를 Allgather하여 remote_labels로 동기화
        int local_offset = 0;
        for (int i = 0; i < mpi_rank; ++i) local_offset += phase1_metrics.partition_vertex_counts[i];
        std::vector<int> all_part_sizes = phase1_metrics.partition_vertex_counts;
        std::vector<int> all_displs(num_partitions, 0);
        for (int i = 1; i < num_partitions; ++i) all_displs[i] = all_displs[i-1] + all_part_sizes[i-1];
        // vertex_labels는 내 파티션 노드만, remote_labels는 전체
        MPI_Allgatherv(vertex_labels.data(), vertex_labels.size(), MPI_INT,
                      remote_labels.data(), all_part_sizes.data(), all_displs.data(), MPI_INT, MPI_COMM_WORLD);

        calculateRatios(mpi_rank, num_partitions, local_graph, vertex_labels, PI);
        calculateImbalance(mpi_rank, num_partitions, PI);
        calculateEdgeCutAndExtractBoundary(mpi_rank, mpi_size, local_graph, vertex_labels, PI, current_edge_cut, previous_edge_cut, edge_rate, BV, NV, PU);
        performDynamicLabelPropagation(mpi_rank, local_graph, vertex_labels, PI, BV, PU, remote_labels, local_offset);
        exchangePartitionUpdates(mpi_rank, mpi_size, BV, NV, PU);
        converged = checkConvergence(mpi_rank, convergence_count, current_edge_cut, previous_edge_cut);
        ++iter;
    }
    double t_end = MPI_Wtime();
    long execution_time_ms = static_cast<long>((t_end - t_start) * 1000);

    // results.cpp의 printFinalResults 호출 (상세 비교/출력)
    extern void printFinalResults(
        int mpi_rank,
        const Phase1Metrics& phase1_metrics,
        int current_edge_cut,
        const std::vector<PartitionInfo>& PI,
        int num_partitions,
        long execution_time_ms
    );
    printFinalResults(mpi_rank, phase1_metrics, current_edge_cut, PI, num_partitions, execution_time_ms);
}
