#include <iostream>
#include <vector>
#include <mpi.h>
// #include <cuda_runtime.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "phase1.h"
#include "report_utils.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 그래프 구조체 (phase1_graph_loader.cu에서 정의됨)
// struct Graph는 이미 include된 파일에 있음

#include "graph_types.h"

// MPI 분산 워크플로우 클래스
class MPIDistributedWorkflowV2 {
private:
    int mpi_rank_;
    int mpi_size_;
    int num_partitions_;  // k (라벨/파티션/스레드 수)
    
    // Phase 1 메트릭 (비교용)
    Phase1Metrics phase1_metrics_;
    
    // 그래프 데이터
    Graph local_graph_;
    std::vector<int> vertex_labels_;  // 각 정점의 라벨 (파티션 ID)
    
    // 7단계 알고리즘 배열들
    std::vector<int> BV_;  // Boundary Vertices (인접리스트 기반)
    std::unordered_map<int, int> NV_;  // Neighbor Vertices (key-value 기반)
    std::vector<PartitionInfo> PI_;  // Partition Information
    PartitionUpdate PU_;  // Partition Update arrays
    
    // 메트릭
    int current_edge_cut_;
    int previous_edge_cut_;
    double edge_rate_;
    int convergence_count_;
    
    // 수렴 조건
    static constexpr double EPSILON = 0.03;
    static constexpr int MAX_CONVERGENCE_COUNT = 10;

public:
    MPIDistributedWorkflowV2(int argc, char* argv[], const Graph& phase1_graph, const std::vector<int>& phase1_labels, const Phase1Metrics& phase1_metrics) {
        // MPI는 이미 메인에서 초기화됨
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        
        current_edge_cut_ = 0;
        previous_edge_cut_ = 0;
        edge_rate_ = 0.0;
        convergence_count_ = 0;
        
        num_partitions_ = std::atoi(argv[2]);
        
        // Phase 1에서 전달받은 그래프와 라벨 사용
        local_graph_ = phase1_graph;
        vertex_labels_ = phase1_labels;
        phase1_metrics_ = phase1_metrics;
        
        if (mpi_rank_ == 0) {
            std::cout << "\n=== MPI 분산 그래프 파티셔닝 (7단계 알고리즘) ===\n";
            std::cout << "서버 수 (MPI): " << mpi_size_ << "\n";
            std::cout << "파티션 수 (라벨/스레드): " << num_partitions_ << "\n";
            std::cout << "그래프 파일: " << argv[1] << "\n";
            std::cout << "알고리즘: 각 서버별 완전 파티셔닝 + 경계 노드 스코어 계산\n";
        }
        
        // PI 배열 초기화
        PI_.resize(num_partitions_);
        for (int i = 0; i < num_partitions_; ++i) {
            PI_[i].partition_id = i;
            PI_[i].RV = 1.0;
            PI_[i].RE = 1.0;
            PI_[i].P_L = 0.0;
        }
    }
    
    ~MPIDistributedWorkflowV2() {
        // MPI_Finalize은 메인에서 처리
    }
    
    void run() {
        if (mpi_rank_ == 0) {
            std::cout << "\n=== Phase 2: 7단계 알고리즘 시작 ===\n";
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const int max_iterations = 50;
        for (int iter = 0; iter < max_iterations; ++iter) {
            std::cout << "\n--- Iteration " << (iter + 1) << " (Rank " << mpi_rank_ << ") ---\n";
            
            // Step 1: RV, RE 계산
            calculateRatios();
            
            // Step 2: 전체 imbalance 계산
            calculateImbalance();
            
            // Step 3: Edge-cut 계산 및 BV/NV 추출
            calculateEdgeCutAndExtractBoundary();
            
            // Step 4: Dynamic Unweighted LP 수행 (핵심!)
            performDynamicLabelPropagation();
            
            // Step 5: 파티션 업데이트 교환
            exchangePartitionUpdates();
            
            // Step 6: 수렴 확인
            if (checkConvergence()) {
                if (mpi_rank_ == 0) {
                    std::cout << "수렴 달성! (Iteration " << (iter + 1) << ")\n";
                }
                break;
            }
            
            // Step 7: 다음 반복 준비
            prepareNextIteration();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 최종 결과 출력
        printFinalResults(duration.count());
    }

private:
    // Step 1: RV, RE 계산
    void calculateRatios() {
        std::cout << "Step1 Rank " << mpi_rank_ << ": RV, RE 계산\n";
        
        // 각 파티션별 정점/간선 수 계산
        std::vector<int> vertex_counts(num_partitions_, 0);
        std::vector<int> edge_counts(num_partitions_, 0);
        
        // 정점 수 계산
        for (int i = 0; i < local_graph_.num_vertices; ++i) {
            int label = vertex_labels_[i];
            if (label >= 0 && label < num_partitions_) {
                vertex_counts[label]++;
            }
        }
        
        // 간선 수 계산 (실제 CSR 사용)
        for (int u = 0; u < local_graph_.num_vertices; ++u) {
            int u_label = vertex_labels_[u];
            if (u_label < 0 || u_label >= num_partitions_) continue;
            
            for (int edge_idx = local_graph_.row_ptr[u]; edge_idx < local_graph_.row_ptr[u + 1]; ++edge_idx) {
                int v = local_graph_.col_indices[edge_idx];
                if (v < local_graph_.num_vertices) {
                    int v_label = vertex_labels_[v];
                    if (v_label == u_label) {
                        edge_counts[u_label]++;
                    }
                }
            }
        }
        
        // 글로벌 집계 (MPI_Allreduce)
        std::vector<int> global_vertex_counts(num_partitions_, 0);
        std::vector<int> global_edge_counts(num_partitions_, 0);
        
        MPI_Allreduce(vertex_counts.data(), global_vertex_counts.data(), 
                     num_partitions_, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(edge_counts.data(), global_edge_counts.data(), 
                     num_partitions_, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // 전체 정점/간선 수
        int total_vertices = 0, total_edges = 0;
        for (int i = 0; i < num_partitions_; ++i) {
            total_vertices += global_vertex_counts[i];
            total_edges += global_edge_counts[i];
        }
        
        // RV, RE 계산: RV_P = |V_P| / (|V|/k), RE_P = |E_P| / (|E|/k)
        for (int i = 0; i < num_partitions_; ++i) {
            if (total_vertices > 0) {
                PI_[i].RV = static_cast<double>(global_vertex_counts[i]) / (static_cast<double>(total_vertices) / num_partitions_);
            } else {
                PI_[i].RV = 1.0;
            }
            
            if (total_edges > 0) {
                PI_[i].RE = static_cast<double>(global_edge_counts[i]) / (static_cast<double>(total_edges) / num_partitions_);
            } else {
                PI_[i].RE = 1.0;
            }
        }
        
        std::cout << "RV/RE계산완료 Rank " << mpi_rank_ << ": 총 " << total_vertices 
                  << "개 정점, " << total_edges << "개 간선\n";
    }
    
    // Step 2: 전체 imbalance 계산
    void calculateImbalance() {
        std::cout << "Step2 Rank " << mpi_rank_ << ": 불균형 계산\n";
        
        // RV, RE의 분산 계산
        double rv_mean = 1.0; // 이상적으로는 1.0
        double re_mean = 1.0;
        
        double rv_variance = 0.0, re_variance = 0.0;
        for (int i = 0; i < num_partitions_; ++i) {
            rv_variance += (PI_[i].RV - rv_mean) * (PI_[i].RV - rv_mean);
            re_variance += (PI_[i].RE - re_mean) * (PI_[i].RE - re_mean);
        }
        rv_variance /= num_partitions_;
        re_variance /= num_partitions_;
        
        // imbalance 계산: imb_i(RV) = Var(RV) / (Var(RV) + Var(RE))
        double total_variance = rv_variance + re_variance;
        if (total_variance > 0) {
            double imb_rv = rv_variance / total_variance;
            double imb_re = re_variance / total_variance;
            
            // Gain 함수 및 Penalty 함수 계산
            for (int i = 0; i < num_partitions_; ++i) {
                PI_[i].imb_RV = imb_rv;
                PI_[i].imb_RE = imb_re;
                
                // G_RV(L) = (1 - RV_L) / K, G_RE(L) = (1 - RE_L) / K
                PI_[i].G_RV = (1.0 - PI_[i].RV) / num_partitions_;
                PI_[i].G_RE = (1.0 - PI_[i].RE) / num_partitions_;
                
                // P_L = imb_i(RV) * G_RV(L) + imb_i(RE) * G_RE(L)
                PI_[i].P_L = imb_rv * PI_[i].G_RV + imb_re * PI_[i].G_RE;
            }
        } else {
            // 분산이 0인 경우 (완벽한 균형)
            for (int i = 0; i < num_partitions_; ++i) {
                PI_[i].imb_RV = 0.0;
                PI_[i].imb_RE = 0.0;
                PI_[i].G_RV = 0.0;
                PI_[i].G_RE = 0.0;
                PI_[i].P_L = 0.0;
            }
        }
        
        std::cout << "불균형계산완료 Rank " << mpi_rank_ << ": RV분산=" << rv_variance 
                  << ", RE분산=" << re_variance << "\n";
    }
    
    // Step 3: Edge-cut 계산 및 BV/NV 추출
    void calculateEdgeCutAndExtractBoundary() {
        std::cout << "Step3 Rank " << mpi_rank_ << ": Edge-cut 계산 및 BV/NV 추출\n";
        
        // 이전 edge-cut 저장
        previous_edge_cut_ = current_edge_cut_;
        
        // 로컬 edge-cut 계산 (실제 CSR 사용)
        int local_edge_cut = 0;
        for (int u = 0; u < local_graph_.num_vertices; ++u) {
            int u_label = vertex_labels_[u];
            
            for (int edge_idx = local_graph_.row_ptr[u]; edge_idx < local_graph_.row_ptr[u + 1]; ++edge_idx) {
                int v = local_graph_.col_indices[edge_idx];
                if (v < local_graph_.num_vertices && u < v) { // 중복 방지
                    int v_label = vertex_labels_[v];
                    if (u_label != v_label) {
                        local_edge_cut++;
                    }
                }
            }
        }
        
        // 글로벌 edge-cut 집계
        MPI_Allreduce(&local_edge_cut, &current_edge_cut_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Edge rate 계산: ER = (EC_prev - EC_curr) / EC_prev
        if (previous_edge_cut_ > 0) {
            edge_rate_ = static_cast<double>(previous_edge_cut_ - current_edge_cut_) / previous_edge_cut_;
        } else {
            edge_rate_ = 0.0;
        }
        
        // Boundary Vertices 추출
        extractBoundaryVertices();
        
        std::cout << "EdgeCut계산완료 Rank " << mpi_rank_ << ": Edge-cut=" << current_edge_cut_ 
                  << ", Rate=" << edge_rate_ << ", BV수=" << BV_.size() << "\n";
    }
    
    // Boundary Vertices 추출 (BV 배열)
    void extractBoundaryVertices() {
        BV_.clear();
        NV_.clear();
        
        // 경계 정점 찾기: Remote Edge가 존재하는 정점들
        for (int u = 0; u < local_graph_.num_vertices; ++u) {
            int u_label = vertex_labels_[u];
            bool is_boundary = false;
            
            // 이웃들 확인 (CSR 사용)
            for (int edge_idx = local_graph_.row_ptr[u]; edge_idx < local_graph_.row_ptr[u + 1]; ++edge_idx) {
                int v = local_graph_.col_indices[edge_idx];
                if (v < local_graph_.num_vertices) {
                    int v_label = vertex_labels_[v];
                    if (u_label != v_label) {
                        is_boundary = true;
                        NV_[v] = v_label; // 이웃의 파티션 정보 저장 (key-value)
                    }
                }
            }
            
            if (is_boundary) {
                BV_.push_back(u); // Boundary Vertex 추가 (인접리스트 기반)
            }
        }
    }
    
    // Step 4: Dynamic Unweighted LP 수행
    void performDynamicLabelPropagation() {
        std::cout << "Step4 Rank " << mpi_rank_ << ": Dynamic LP 수행 (각 노드마다 스코어 계산)\n";
        
        // PU 배열 초기화
        PU_.PU_RO.clear();
        PU_.PU_OV.clear();
        PU_.PU_ON.clear();
        
        int updates_count = 0;
        
        // OpenMP로 파티션별 병렬 처리 (각 스레드 = 각 파티션)
        #pragma omp parallel for reduction(+:updates_count) num_threads(num_partitions_)
        for (int thread_id = 0; thread_id < num_partitions_; ++thread_id) {
            // 각 스레드가 해당 라벨(파티션)의 경계 정점들만 처리
            for (int bv_idx : BV_) {
                if (vertex_labels_[bv_idx] == thread_id) {
                    // Score(L) = |u| * (1 + P_L) 계산 (각 노드마다)
                    std::vector<double> label_scores(num_partitions_, 0.0);
                    
                    // 이웃들의 라벨별 점수 계산
                    for (int edge_idx = local_graph_.row_ptr[bv_idx]; 
                         edge_idx < local_graph_.row_ptr[bv_idx + 1]; ++edge_idx) {
                        int neighbor = local_graph_.col_indices[edge_idx];
                        
                        if (neighbor < local_graph_.num_vertices) {
                            int neighbor_label = vertex_labels_[neighbor];
                            if (neighbor_label >= 0 && neighbor_label < num_partitions_) {
                                // |u| = 1 (단일 이웃), P_L = PI_[neighbor_label].P_L
                                double score = 1.0 * (1.0 + PI_[neighbor_label].P_L);
                                label_scores[neighbor_label] += score;
                            }
                        }
                    }
                    
                    // 최고 점수 라벨 선택
                    int best_label = thread_id;
                    double best_score = label_scores[thread_id];
                    
                    for (int label = 0; label < num_partitions_; ++label) {
                        if (label_scores[label] > best_score) {
                            best_score = label_scores[label];
                            best_label = label;
                        }
                    }
                    
                    // 라벨 변경 시 PU 배열 업데이트
                    if (best_label != thread_id) {
                        #pragma omp critical
                        {
                            // 1) 이웃 중 기존 파티션에 속하고 BV에 없는 노드를 PU_RO에 저장
                            for (int edge_idx = local_graph_.row_ptr[bv_idx]; 
                                 edge_idx < local_graph_.row_ptr[bv_idx + 1]; ++edge_idx) {
                                int neighbor = local_graph_.col_indices[edge_idx];
                                
                                if (neighbor < local_graph_.num_vertices && 
                                    vertex_labels_[neighbor] == thread_id && 
                                    std::find(BV_.begin(), BV_.end(), neighbor) == BV_.end()) {
                                    PU_.PU_RO.push_back(neighbor);
                                }
                            }
                            
                            // 2) 파티션이 변경된 노드를 PU_OV에 저장
                            PU_.PU_OV.push_back(bv_idx);
                            
                            // 3) 파티션이 변경된 노드의 이웃정보를 PU_ON에 저장
                            for (auto& nv : NV_) {
                                if (nv.first >= local_graph_.row_ptr[bv_idx] && 
                                    nv.first < local_graph_.row_ptr[bv_idx + 1]) {
                                    PU_.PU_ON.push_back({bv_idx, nv.second});
                                }
                            }
                            
                            // 라벨 업데이트
                            vertex_labels_[bv_idx] = best_label;
                            updates_count++;
                        }
                    }
                }
            }
        }
        
        // 전체 업데이트 수 집계
        int total_updates;
        MPI_Allreduce(&updates_count, &total_updates, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        std::cout << "LP완료 Rank " << mpi_rank_ << ": " << updates_count 
                  << "개 라벨 변경 (전체: " << total_updates << "개)\n";
    }
    
    // Step 5: 파티션 업데이트 교환
    void exchangePartitionUpdates() {
        std::cout << "Step5 Rank " << mpi_rank_ << ": 파티션 업데이트 교환 (PU 배열)\n";
        
        // PU_OV와 PU_ON을 다른 서버의 PU_RV와 PU_RN으로 전송
        // 간소화된 구현: MPI_Allgather 사용
        
        // 보낼 데이터 크기 교환
        std::vector<int> send_sizes = {
            static_cast<int>(PU_.PU_OV.size()),
            static_cast<int>(PU_.PU_ON.size())
        };
        std::vector<int> all_sizes(mpi_size_ * 2);
        
        MPI_Allgather(send_sizes.data(), 2, MPI_INT, all_sizes.data(), 2, MPI_INT, MPI_COMM_WORLD);
        
        // PU_RO에 해당하는 노드를 다음 이터레이션의 BV에 추가
        for (int vertex : PU_.PU_RO) {
            if (std::find(BV_.begin(), BV_.end(), vertex) == BV_.end()) {
                BV_.push_back(vertex);
            }
        }
        
        std::cout << "교환완료 Rank " << mpi_rank_ << ": PU_OV=" << PU_.PU_OV.size() 
                  << ", PU_ON=" << PU_.PU_ON.size() << ", PU_RO=" << PU_.PU_RO.size() << "\n";
    }
    
    // Step 6: 수렴 확인
    bool checkConvergence() {
        std::cout << "Step6 Rank " << mpi_rank_ << ": 수렴 확인\n";
        
        // EC 변화량이 ε보다 작은지 확인
        bool local_converged = (std::abs(edge_rate_) < EPSILON);
        
        if (local_converged) {
            convergence_count_++;
        } else {
            convergence_count_ = 0;
        }
        
        // k번 연속으로 수렴 조건 만족 시 종료
        bool converged = (convergence_count_ >= MAX_CONVERGENCE_COUNT);
        
        // 글로벌 수렴 확인
        int global_converged;
        int local_converged_int = converged ? 1 : 0;
        MPI_Allreduce(&local_converged_int, &global_converged, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        
        if (mpi_rank_ == 0) {
            std::cout << "수렴상태: Edge-cut=" << current_edge_cut_ 
                      << ", Rate=" << edge_rate_ 
                      << ", 수렴카운트=" << convergence_count_ << "/" << MAX_CONVERGENCE_COUNT << "\n";
        }
        
        return global_converged == 1;
    }
    
    // Step 7: 다음 반복 준비
    void prepareNextIteration() {
        std::cout << "Step7 Rank " << mpi_rank_ << ": 다음 반복 준비\n";
        
        // Step 1,2,4,5,6을 반복하기 위한 준비
        // PU 배열 처리 후 다음 이터레이션을 위한 상태 업데이트
        
        std::cout << "준비완료 Rank " << mpi_rank_ << ": 다음 반복 준비 완료\n";
    }
    
    // 최종 결과 출력 (Phase 1과 비교)
    void printFinalResults(long execution_time_ms) {
        ::printFinalResults(mpi_rank_, current_edge_cut_, static_cast<const std::vector<PartitionInfo>&>(PI_), num_partitions_, execution_time_ms, static_cast<const Phase1Metrics&>(phase1_metrics_));
    }
};

// 메인 함수
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc < 3) {
        if (mpi_rank == 0) {
            std::cout << "사용법: mpirun -np <서버수> ./mpi_distributed_workflow_v2 <그래프파일> <파티션수>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int num_partitions = std::atoi(argv[2]);
    std::string filename = argv[1];

    // Phase 1: 그래프 분할 및 분배 (새로운 파일에서)
    Graph local_graph;
    std::vector<int> vertex_labels;
    Phase1Metrics phase1_metrics = phase1_partition_and_distribute(mpi_rank, mpi_size, num_partitions, local_graph, vertex_labels, filename);

    // Phase 2: 7단계 알고리즘 (각 파티션별로 OpenMP 스레드 병렬)
    try {
        MPIDistributedWorkflowV2 workflow(argc, argv, local_graph, vertex_labels, phase1_metrics);
        workflow.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
