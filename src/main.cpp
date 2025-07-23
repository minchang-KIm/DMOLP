#include <iostream>
#include <vector>
#include <mpi.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cuda_kernels.h"
#endif
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <mutex>
#include "phase1.h"

#ifdef USE_CUDA
#include "cuda_kernels.h"
#include "cuda_kernels.cu"
#else
#define CUDA_CHECK(call) // CUDA 비활성화 시 빈 매크로
#endif

// MPI 분산 워크플로우 클래스
class MPIDistributedWorkflowV2 {
private:
    int mpi_rank_;
    int mpi_size_;
    int num_partitions_;
    
    // 정점 소유권 정보
    int start_vertex_;  // 이 MPI 프로세스가 소유하는 정점 시작 ID
    int end_vertex_;    // 이 MPI 프로세스가 소유하는 정점 끝 ID (exclusive)  // k (라벨/파티션/스레드 수)
    
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
    static constexpr double EPSILON = 0.005;  // 더 엄격한 수렴 조건
    static constexpr int MAX_CONVERGENCE_COUNT = 5;  // 연속 수렴 횟수 줄임
    
#ifdef USE_CUDA
    // GPU 메모리 매니저 (Multi-GPU 지원)
    std::vector<std::unique_ptr<GPUMemoryManager>> gpu_managers_;
    int num_gpus_;
    std::vector<cudaDeviceProp> gpu_properties_;
#endif

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
        
        // 정점 소유권 계산 (Phase 1과 동일한 로직)
        int vertices_per_rank = phase1_metrics.total_vertices / mpi_size_;
        start_vertex_ = mpi_rank_ * vertices_per_rank;
        end_vertex_ = (mpi_rank_ == mpi_size_ - 1) ? phase1_metrics.total_vertices : (mpi_rank_ + 1) * vertices_per_rank;
        
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
        
#ifdef USE_CUDA
        // === Multi-GPU 동적 감지 및 초기화 ===
        // 1. 현재 서버의 GPU 개수 확인
        cudaGetDeviceCount(&num_gpus_);
        
        if (num_gpus_ <= 0) {
            std::cerr << "Rank " << mpi_rank_ << ": GPU를 찾을 수 없습니다!\n";
            num_gpus_ = 0;
        } else {
            std::cout << "Rank " << mpi_rank_ << ": " << num_gpus_ << "개 GPU 감지\n";
            
            // 2. 각 GPU 정보 수집
            gpu_properties_.resize(num_gpus_);
            gpu_managers_.resize(num_gpus_);
            
            for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
                cudaGetDeviceProperties(&gpu_properties_[gpu_id], gpu_id);
                
                // GPU 상세 정보 출력
                std::cout << "  GPU " << gpu_id << ": " << gpu_properties_[gpu_id].name 
                          << " (메모리: " << gpu_properties_[gpu_id].totalGlobalMem / (1024*1024*1024) << "GB, "
                          << "코어: " << gpu_properties_[gpu_id].multiProcessorCount << "개)\n";
                
                // 각 GPU별 메모리 매니저 초기화
                cudaSetDevice(gpu_id);
                
                size_t num_edges = 0;
                if (local_graph_.num_vertices > 0) {
                    num_edges = local_graph_.row_ptr[local_graph_.num_vertices];
                }
                
                gpu_managers_[gpu_id] = std::make_unique<GPUMemoryManager>(
                    local_graph_.num_vertices, num_edges);
                
                // 그래프 데이터를 각 GPU로 복사
                gpu_managers_[gpu_id]->copyToGPU(vertex_labels_, local_graph_.row_ptr, local_graph_.col_indices);
            }
            
            std::cout << "Rank " << mpi_rank_ << ": Multi-GPU 초기화 완료 (" << num_gpus_ << "개 GPU)\n";
        }
        
        if (mpi_rank_ == 0) {
            std::cout << "Multi-GPU 가속 활성화: CUDA + OpenMP 하이브리드 + 동적 GPU 할당\n";
        }
#endif
    }
    
    ~MPIDistributedWorkflowV2() {
        // MPI_Finalize은 메인에서 처리
    }
    
    void run() {
        if (mpi_rank_ == 0) {
            std::cout << "\n=== Phase 2: 7단계 알고리즘 시작 ===\n";
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const int max_iterations = 100;  // 적당한 반복 횟수
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
    // CUDA 상수 메모리에 파티션 정보 복사 함수
#ifdef USE_CUDA
    void updatePartitionInfoOnGPU() {
        if (PI_.size() > 16) {
            std::cerr << "Error: 파티션 수가 16개를 초과했습니다!" << std::endl;
            return;
        }
        
        // CPU 메모리에 GPU용 파티션 정보 준비
        PartitionInfoGPU gpu_partition_info[16];
        for (size_t i = 0; i < PI_.size(); ++i) {
            gpu_partition_info[i].partition_id = PI_[i].partition_id;
            gpu_partition_info[i].RV = PI_[i].RV;
            gpu_partition_info[i].RE = PI_[i].RE;
            gpu_partition_info[i].P_L = PI_[i].P_L;
        }
        
        // GPU 상수 메모리로 복사
        CUDA_CHECK(cudaMemcpyToSymbol(d_partition_info, gpu_partition_info, 
                                      PI_.size() * sizeof(PartitionInfoGPU)));
    }
#else
    void updatePartitionInfoOnGPU() {
        // CPU 모드에서는 아무것도 하지 않음
    }
#endif

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
        
        int local_edge_cut = 0;
        
#ifdef USE_CUDA
        // Multi-GPU로 Edge-cut 계산
        std::cout << "  [Multi-GPU] " << num_gpus_ << "개 GPU로 CUDA Edge-cut 계산 중...\n";
        
        if (num_gpus_ > 0) {
            std::vector<int> gpu_edge_cuts(num_gpus_, 0);
            
            // GPU별 병렬 Edge-cut 계산
            #pragma omp parallel for
            for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
                cudaSetDevice(gpu_id);
                gpu_edge_cuts[gpu_id] = gpu_managers_[gpu_id]->calculateEdgeCut();
            }
            
            // GPU 결과 합산 (중복 제거 로직)
            for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
                local_edge_cut += gpu_edge_cuts[gpu_id];
                std::cout << "    GPU " << gpu_id << ": " << gpu_edge_cuts[gpu_id] << " edge-cut\n";
            }
            
            // 중복 계산 방지를 위해 GPU 개수로 나누기
            local_edge_cut = local_edge_cut / num_gpus_;
            
            std::cout << "  [Multi-GPU] Edge-cut 계산 완료: " << local_edge_cut << "\n";
        } else {
            // GPU 없는 경우 CPU 계산
            std::cout << "  [CPU Fallback] OpenMP Edge-cut 계산 중...\n";
            
            #pragma omp parallel for reduction(+:local_edge_cut) schedule(dynamic, 1000)
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
            std::cout << "  [CPU Fallback] Edge-cut 계산 완료: " << local_edge_cut << "\n";
        }
#else
        // CPU OpenMP로 Edge-cut 계산
        std::cout << "  [CPU] OpenMP Edge-cut 계산 중...\n";
        
        #pragma omp parallel for reduction(+:local_edge_cut) schedule(dynamic, 1000)
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
        std::cout << "  [CPU] Edge-cut 계산 완료: " << local_edge_cut << "\n";
#endif
        
        // 글로벌 edge-cut 집계
        MPI_Allreduce(&local_edge_cut, &current_edge_cut_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Edge rate 계산: ER = (EC_prev - EC_curr) / EC_prev
        if (previous_edge_cut_ > 0) {
            edge_rate_ = static_cast<double>(previous_edge_cut_ - current_edge_cut_) / previous_edge_cut_;
        } else {
            edge_rate_ = 0.0;
        }
        
        // Boundary Vertices 추출 (OpenMP 활용)
        extractBoundaryVerticesWithOpenMP();
        
        std::cout << "EdgeCut계산완료 Rank " << mpi_rank_ << ": Edge-cut=" << current_edge_cut_ 
                  << ", Rate=" << edge_rate_ << ", BV수=" << BV_.size() << "\n";
    }
    
    // Boundary Vertices 추출 (OpenMP 최적화)
    void extractBoundaryVerticesWithOpenMP() {
        BV_.clear();
        NV_.clear();
        
        // OpenMP로 파티션별 병렬 경계 정점 추출
        std::vector<std::vector<int>> thread_boundary_vertices(omp_get_max_threads());
        std::vector<std::unordered_map<int, int>> thread_neighbor_vertices(omp_get_max_threads());
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            
            #pragma omp for schedule(dynamic, 1000)
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
                            thread_neighbor_vertices[thread_id][v] = v_label; // 이웃의 파티션 정보 저장
                        }
                    }
                }
                
                if (is_boundary) {
                    thread_boundary_vertices[thread_id].push_back(u); // Boundary Vertex 추가
                }
            }
        }
        
        // 스레드별 결과를 병합
        for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
            BV_.insert(BV_.end(), thread_boundary_vertices[tid].begin(), thread_boundary_vertices[tid].end());
            for (auto& pair : thread_neighbor_vertices[tid]) {
                NV_[pair.first] = pair.second;
            }
        }
    }
    
    // 기존 Boundary Vertices 추출 (호환성 유지)
    void extractBoundaryVertices() {
        extractBoundaryVerticesWithOpenMP();
    }
    
    // Step 4: Dynamic Unweighted LP 수행 (핵심!)
    void performDynamicLabelPropagation() {
        std::cout << "Step4 Rank " << mpi_rank_ << ": Dynamic LP 수행 ";
        
        // PU 배열 초기화
        PU_.PU_RO.clear();
        PU_.PU_OV.clear();
        PU_.PU_ON.clear();
        
        int updates_count = 0;

#ifdef USE_CUDA
        std::cout << "(CUDA GPU 가속)\n";
        updates_count = performDynamicLabelPropagationGPU();
#else
        std::cout << "(CPU OpenMP)\n";
        updates_count = performDynamicLabelPropagationCPU();
#endif
        
        // 전체 업데이트 수 집계
        int total_updates;
        MPI_Allreduce(&updates_count, &total_updates, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        std::cout << "LP완료 Rank " << mpi_rank_ << ": " << updates_count 
                  << "개 라벨 변경 (전체: " << total_updates << "개)\n";
    }
    
#ifdef USE_CUDA
    // Multi-GPU 구현 (스레드 풀 + GPU 큐 방식)
    int performDynamicLabelPropagationGPU() {
        std::cout << "  [Multi-GPU] " << num_gpus_ << "개 GPU + OpenMP 스레드 풀 (파티션별 처리)\n";
        
        if (num_gpus_ <= 0) {
            std::cout << "  [Multi-GPU] GPU 없음, CPU 모드로 fallback\n";
            return performDynamicLabelPropagationCPU();
        }
        
        if (BV_.empty()) {
            std::cout << "  [Multi-GPU] 경계 정점 없음, 라벨 변경 없음\n";
            return 0;
        }
        
        std::cout << "  [Multi-GPU] " << BV_.size() << "개 경계 정점을 " << num_partitions_ << "개 파티션별로 처리\n";
        
        // === 파티션별 경계 정점 분류 ===
        std::vector<std::vector<int>> partition_boundary_vertices(num_partitions_);
        
        // 경계 정점을 파티션별로 분류
        for (int vertex : BV_) {
            int partition_id = vertex_labels_[vertex];
            if (partition_id >= 0 && partition_id < num_partitions_) {
                partition_boundary_vertices[partition_id].push_back(vertex);
            }
        }
        
        // 각 파티션별 경계 정점 수 출력
        for (int partition_id = 0; partition_id < num_partitions_; ++partition_id) {
            std::cout << "    파티션 " << partition_id << ": " << partition_boundary_vertices[partition_id].size() 
                      << "개 경계 정점\n";
        }
        
        // 파티션 정보를 GPU 형식으로 변환
        std::vector<PartitionInfoGPU> gpu_partition_info(num_partitions_);
        for (int i = 0; i < num_partitions_; ++i) {
            gpu_partition_info[i].partition_id = PI_[i].partition_id;
            gpu_partition_info[i].RV = PI_[i].RV;
            gpu_partition_info[i].RE = PI_[i].RE;
            gpu_partition_info[i].P_L = PI_[i].P_L;
        }
        
        // GPU 상수 메모리에 파티션 정보 복사
        updatePartitionInfoOnGPU();
        
        // === 스레드 풀 + GPU 큐 방식 (파티션별 처리) ===
        std::vector<int> partition_updates(num_partitions_, 0);
        std::atomic<int> next_gpu{0}; // GPU 큐 인덱스
        std::vector<std::mutex> gpu_mutexes(num_gpus_); // GPU별 뮤텍스
        
        // OpenMP 스레드 수를 파티션 수로 설정
        omp_set_num_threads(num_partitions_);
        
        #pragma omp parallel for
        for (int partition_id = 0; partition_id < num_partitions_; ++partition_id) {
            if (partition_boundary_vertices[partition_id].empty()) continue;
            
            int thread_id = omp_get_thread_num();
            
            // 스레드별 GPU 동적 할당 (Round-robin)
            int assigned_gpu = next_gpu.fetch_add(1) % num_gpus_;
            
            // GPU별 동기화 보장
            {
                std::lock_guard<std::mutex> lock(gpu_mutexes[assigned_gpu]);
                
                // GPU 컨텍스트 설정
                cudaSetDevice(assigned_gpu);
                
                std::cout << "    스레드 " << thread_id << " (파티션 " << partition_id 
                          << ") → GPU " << assigned_gpu << " 할당\n";
                
                // 해당 파티션의 경계 정점들만 처리
                int updates = gpu_managers_[assigned_gpu]->performDynamicLabelPropagation(
                    partition_boundary_vertices[partition_id], gpu_partition_info, num_partitions_, mpi_rank_, start_vertex_, end_vertex_);
                
                partition_updates[partition_id] = updates;
                
                std::cout << "    스레드 " << thread_id << " (파티션 " << partition_id 
                          << ", GPU " << assigned_gpu << "): " << updates << "개 라벨 변경\n";
            }
        }
        
        // === 모든 파티션 결과 집계 ===
        std::cout << "  [Multi-GPU] 파티션별 결과 집계 중...\n";
        
        int total_updates = 0;
        
        // GPU 동기화
        for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
            cudaSetDevice(gpu_id);
            cudaDeviceSynchronize();
        }
        
        // 파티션별 업데이트 집계
        for (int partition_id = 0; partition_id < num_partitions_; ++partition_id) {
            total_updates += partition_updates[partition_id];
            
            if (partition_updates[partition_id] > 0) {
                std::cout << "    파티션 " << partition_id << ": " 
                          << partition_updates[partition_id] << "개 라벨 변경\n";
            }
        }
        
        // 라벨 변경사항을 CPU로 복사 (GPU 0번에서 전체 복사)
        if (total_updates > 0) {
            cudaSetDevice(0);
            gpu_managers_[0]->copyToCPU(vertex_labels_);
            std::cout << "    GPU 0에서 전체 라벨 데이터 CPU로 복사 완료\n";
        }
        
        std::cout << "  [Multi-GPU] 총 " << total_updates << "개 라벨 변경 완료\n";
        return total_updates;
    }
#endif
    
    // CPU OpenMP 구현 (올바른 로직)
    int performDynamicLabelPropagationCPU() {
        std::cout << "  [CPU] OpenMP 병렬 처리 시작\n";
        
        if (BV_.empty()) {
            std::cout << "  [CPU] 경계 정점 없음, 라벨 변경 없음\n";
            return 0;
        }
        
        std::cout << "  [CPU] " << BV_.size() << "개 경계 정점에서 Label Propagation 수행\n";
        
        int updates_count = 0;
        std::vector<std::pair<int, int>> label_changes; // {vertex, new_label}
        
        // OpenMP로 경계 정점들 병렬 처리
        #pragma omp parallel
        {
            std::vector<std::pair<int, int>> thread_label_changes;
            
            #pragma omp for schedule(dynamic, 100)
            for (int i = 0; i < BV_.size(); ++i) {
                int vertex = BV_[i];
                int current_label = vertex_labels_[vertex];
                
                // Score(L) = |u| * (1 + P_L) 계산 (각 노드마다)
                std::vector<double> label_scores(num_partitions_, 0.0);
                
                // 이웃들의 라벨별 점수 계산
                for (int edge_idx = local_graph_.row_ptr[vertex]; 
                     edge_idx < local_graph_.row_ptr[vertex + 1]; ++edge_idx) {
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
                int best_label = current_label;
                double best_score = label_scores[current_label];
                
                for (int label = 0; label < num_partitions_; ++label) {
                    if (label_scores[label] > best_score) {
                        best_score = label_scores[label];
                        best_label = label;
                    }
                }
                
                // 라벨 변경이 필요한 경우
                if (best_label != current_label) {
                    thread_label_changes.push_back({vertex, best_label});
                }
            }
            
            // 스레드별 결과를 안전하게 병합
            #pragma omp critical
            {
                label_changes.insert(label_changes.end(), 
                                   thread_label_changes.begin(), thread_label_changes.end());
            }
        }
        
        // 라벨 변경 적용 (순차 처리)
        for (auto& change : label_changes) {
            int vertex = change.first;
            int new_label = change.second;
            int old_label = vertex_labels_[vertex];
            
            // PU 배열 업데이트
            // 1) 이웃 중 기존 파티션에 속하고 BV에 없는 노드를 PU_RO에 저장
            for (int edge_idx = local_graph_.row_ptr[vertex]; 
                 edge_idx < local_graph_.row_ptr[vertex + 1]; ++edge_idx) {
                int neighbor = local_graph_.col_indices[edge_idx];
                
                if (neighbor < local_graph_.num_vertices && 
                    vertex_labels_[neighbor] == old_label && 
                    std::find(BV_.begin(), BV_.end(), neighbor) == BV_.end()) {
                    PU_.PU_RO.push_back(neighbor);
                }
            }
            
            // 2) 파티션이 변경된 노드를 PU_OV에 저장
            PU_.PU_OV.push_back(vertex);
            
            // 3) 파티션이 변경된 노드의 이웃정보를 PU_ON에 저장
            for (auto& nv : NV_) {
                if (nv.first >= local_graph_.row_ptr[vertex] && 
                    nv.first < local_graph_.row_ptr[vertex + 1]) {
                    PU_.PU_ON.push_back({vertex, nv.second});
                }
            }
            
            // 라벨 업데이트
            vertex_labels_[vertex] = new_label;
            updates_count++;
        }
        
        std::cout << "  [CPU] " << updates_count << "개 라벨 변경 완료\n";
        return updates_count;
    }
    
    // Step 5: 파티션 업데이트 교환 (OpenMP 병렬 처리)
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
        
        // 실제 데이터 교환 (간소화)
        // 실제 구현에서는 더 복잡한 MPI 통신 필요
        
        // PU_RO에 해당하는 노드를 다음 이터레이션의 BV에 추가 (OpenMP 병렬)
        std::vector<int> new_boundary_candidates = PU_.PU_RO;
        
        #pragma omp parallel for
        for (int i = 0; i < new_boundary_candidates.size(); ++i) {
            int vertex = new_boundary_candidates[i];
            
            #pragma omp critical
            {
                if (std::find(BV_.begin(), BV_.end(), vertex) == BV_.end()) {
                    BV_.push_back(vertex);
                }
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
    
    // Step 7: 다음 반복 준비 + Ghost Node 복제
    void prepareNextIteration() {
        std::cout << "Step7 Rank " << mpi_rank_ << ": Ghost Node 복제 및 다음 반복 준비\n";
        
        // === Ghost Node 복제 메커니즘 ===
        // 1. 경계 정점의 이웃 정보 수집
        std::vector<std::pair<int, int>> ghost_node_updates; // {vertex_id, new_label}
        
        // 2. MPI 프로세스 간 Ghost Node 정보 교환
        exchangeGhostNodeInformation(ghost_node_updates);
        
        // 3. Ghost Node 정보를 로컬 복제본에 반영
        applyGhostNodeUpdates(ghost_node_updates);
        
        // 4. PU 배열 처리 후 다음 이터레이션을 위한 상태 업데이트
        updatePartitionBoundaries();
        
        std::cout << "준비완료 Rank " << mpi_rank_ << ": Ghost Node 복제 완료, 다음 반복 준비 완료\n";
    }
    
    // Ghost Node 정보 교환 (MPI 통신)
    void exchangeGhostNodeInformation(std::vector<std::pair<int, int>>& ghost_updates) {
        std::cout << "  [Ghost Node] MPI 프로세스 간 Ghost Node 정보 교환 중...\n";
        
        // 각 MPI 프로세스가 소유하지 않는 정점들의 라벨 정보를 수집
        std::vector<int> remote_vertices;  // 다른 프로세스 소유 정점들
        std::vector<int> remote_labels;    // 해당 정점들의 최신 라벨
        
        // 경계 정점들의 이웃 중 다른 프로세스 소유 정점 찾기
        for (int boundary_vertex : BV_) {
            for (int edge_idx = local_graph_.row_ptr[boundary_vertex]; 
                 edge_idx < local_graph_.row_ptr[boundary_vertex + 1]; ++edge_idx) {
                int neighbor = local_graph_.col_indices[edge_idx];
                
                // 이웃이 다른 MPI 프로세스 소유 정점인 경우
                if (neighbor < start_vertex_ || neighbor >= end_vertex_) {
                    remote_vertices.push_back(neighbor);
                }
            }
        }
        
        // 중복 제거
        std::sort(remote_vertices.begin(), remote_vertices.end());
        remote_vertices.erase(std::unique(remote_vertices.begin(), remote_vertices.end()), 
                             remote_vertices.end());
        
        std::cout << "  [Ghost Node] " << remote_vertices.size() << "개 원격 정점의 Ghost Node 정보 요청\n";
        
        // === MPI 통신으로 Ghost Node 정보 교환 ===
        // 모든 프로세스에게 요청하는 정점 ID들을 브로드캐스트
        int num_requests = remote_vertices.size();
        std::vector<int> all_requests;
        std::vector<int> request_counts(mpi_size_);
        
        // 각 프로세스의 요청 개수 수집
        MPI_Allgather(&num_requests, 1, MPI_INT, request_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // 전체 요청 크기 계산
        std::vector<int> request_displs(mpi_size_);
        int total_requests = 0;
        for (int i = 0; i < mpi_size_; ++i) {
            request_displs[i] = total_requests;
            total_requests += request_counts[i];
        }
        all_requests.resize(total_requests);
        
        // 모든 요청 정점 ID 수집
        MPI_Allgatherv(remote_vertices.data(), num_requests, MPI_INT,
                       all_requests.data(), request_counts.data(), request_displs.data(), 
                       MPI_INT, MPI_COMM_WORLD);
        
        // 자신이 소유한 정점들의 라벨 정보 제공
        std::vector<int> response_labels;
        for (int vertex_id : all_requests) {
            if (vertex_id >= start_vertex_ && vertex_id < end_vertex_) {
                int local_index = vertex_id - start_vertex_;
                if (local_index >= 0 && local_index < vertex_labels_.size()) {
                    response_labels.push_back(vertex_labels_[local_index]);
                } else {
                    response_labels.push_back(-1); // 무효한 정점
                }
            } else {
                response_labels.push_back(-1); // 소유하지 않은 정점
            }
        }
        
        // 모든 응답 라벨 수집
        std::vector<int> all_response_labels(total_requests * mpi_size_);
        MPI_Allgather(response_labels.data(), total_requests, MPI_INT,
                      all_response_labels.data(), total_requests, MPI_INT, MPI_COMM_WORLD);
        
        // Ghost Node 업데이트 정보 구성
        for (int i = 0; i < remote_vertices.size(); ++i) {
            int vertex_id = remote_vertices[i];
            
            // 해당 정점을 소유한 프로세스로부터 최신 라벨 정보 획득
            for (int rank = 0; rank < mpi_size_; ++rank) {
                int response_idx = rank * total_requests + request_displs[mpi_rank_] + i;
                if (response_idx < all_response_labels.size()) {
                    int label = all_response_labels[response_idx];
                    if (label >= 0) { // 유효한 라벨
                        ghost_updates.push_back({vertex_id, label});
                        break;
                    }
                }
            }
        }
        
        std::cout << "  [Ghost Node] " << ghost_updates.size() << "개 Ghost Node 업데이트 수신 완료\n";
    }
    
    // Ghost Node 업데이트 적용
    void applyGhostNodeUpdates(const std::vector<std::pair<int, int>>& ghost_updates) {
        std::cout << "  [Ghost Node] 로컬 Ghost Node 복제본에 업데이트 적용 중...\n";
        
        int updates_applied = 0;
        
        // Ghost Node 정보를 NV_ (Neighbor Vertices) 맵에 업데이트
        for (const auto& update : ghost_updates) {
            int vertex_id = update.first;
            int new_label = update.second;
            
            // NV_ 맵에서 해당 정점의 라벨 정보 업데이트
            if (NV_.find(vertex_id) != NV_.end()) {
                NV_[vertex_id] = new_label;
                updates_applied++;
            } else {
                // 새로운 Ghost Node 추가
                NV_[vertex_id] = new_label;
                updates_applied++;
            }
        }
        
        std::cout << "  [Ghost Node] " << updates_applied << "개 Ghost Node 복제본 업데이트 완료\n";
    }
    
    // 파티션 경계 업데이트
    void updatePartitionBoundaries() {
        std::cout << "  [Ghost Node] 파티션 경계 정점 상태 업데이트 중...\n";
        
        // PU_RO에 해당하는 노드를 다음 이터레이션의 BV에 추가
        std::vector<int> new_boundary_candidates = PU_.PU_RO;
        
        int new_boundaries_added = 0;
        
        #pragma omp parallel for reduction(+:new_boundaries_added)
        for (int i = 0; i < new_boundary_candidates.size(); ++i) {
            int vertex = new_boundary_candidates[i];
            
            // 이미 BV에 있는지 확인
            bool already_boundary = false;
            
            #pragma omp critical
            {
                if (std::find(BV_.begin(), BV_.end(), vertex) == BV_.end()) {
                    BV_.push_back(vertex);
                    new_boundaries_added++;
                } else {
                    already_boundary = true;
                }
            }
        }
        
        std::cout << "  [Ghost Node] " << new_boundaries_added << "개 새로운 경계 정점 추가\n";
    }
    
    // 최종 결과 출력 (Phase 1과 비교)
    void printFinalResults(long execution_time_ms) {
        if (mpi_rank_ != 0) return;
        
        std::cout << "\n=== 최종 결과 (7단계 알고리즘) ===\n";
        
        // Edge-cut 평가
        std::cout << "Edge-cut: " << current_edge_cut_ << "\n";
        
        // Vertex/Edge Balance 계산
        double max_vertex_ratio = 0.0, max_edge_ratio = 0.0;
        for (int i = 0; i < num_partitions_; ++i) {
            max_vertex_ratio = std::max(max_vertex_ratio, PI_[i].RV);
            max_edge_ratio = std::max(max_edge_ratio, PI_[i].RE);
        }
        
        // VB = max_i |V_i| / (|V|/k), EB = max_i |E_i| / (|E|/k)
        std::cout << "Vertex Balance (VB): " << max_vertex_ratio << "\n";
        std::cout << "Edge Balance (EB): " << max_edge_ratio << "\n";
        std::cout << "Execution Time: " << execution_time_ms << " ms\n";
        
        std::cout << "\n파티션별 상세 정보:\n";
        for (int i = 0; i < num_partitions_; ++i) {
            std::cout << "Partition " << i << ": RV=" << PI_[i].RV 
                      << ", RE=" << PI_[i].RE 
                      << ", P_L=" << PI_[i].P_L 
                      << ", G_RV=" << PI_[i].G_RV 
                      << ", G_RE=" << PI_[i].G_RE << "\n";
        }
        
        std::cout << "\n평가 메트릭:\n";
        std::cout << "- Edge-cut: " << current_edge_cut_ << " (최소화 목표)\n";
        std::cout << "- Vertex Balance: " << max_vertex_ratio << " (1.0에 가까울수록 좋음)\n";
        std::cout << "- Edge Balance: " << max_edge_ratio << " (1.0에 가까울수록 좋음)\n";
        std::cout << "- 총 실행시간: " << execution_time_ms << " ms\n";
        
        // *** Phase 1 vs Phase 2 비교 ***
        std::cout << "\n=== Phase 1 vs Phase 2 (7단계 알고리즘) 비교 ===\n";
        
        // Edge-cut 개선율
        double edge_cut_improvement = 0.0;
        if (phase1_metrics_.initial_edge_cut > 0) {
            edge_cut_improvement = (static_cast<double>(phase1_metrics_.initial_edge_cut - current_edge_cut_) / phase1_metrics_.initial_edge_cut) * 100.0;
        }
        
        // Vertex Balance 개선율
        double vertex_balance_improvement = 0.0;
        if (phase1_metrics_.initial_vertex_balance > 0) {
            vertex_balance_improvement = ((phase1_metrics_.initial_vertex_balance - max_vertex_ratio) / phase1_metrics_.initial_vertex_balance) * 100.0;
        }
        
        // Edge Balance 개선율
        double edge_balance_improvement = 0.0;
        if (phase1_metrics_.initial_edge_balance > 0) {
            edge_balance_improvement = ((phase1_metrics_.initial_edge_balance - max_edge_ratio) / phase1_metrics_.initial_edge_balance) * 100.0;
        }
        
        std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "│                    메트릭 비교 결과                         │\n";
        std::cout << "├─────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Edge-cut:                                                   │\n";
        std::cout << "│   Phase 1 (초기): " << std::setw(10) << phase1_metrics_.initial_edge_cut << "                              │\n";
        std::cout << "│   Phase 2 (최종): " << std::setw(10) << current_edge_cut_ << "                              │\n";
        std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_cut_improvement << "%                             │\n";
        std::cout << "│                                                             │\n";
        std::cout << "│ Vertex Balance:                                             │\n";
        std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << phase1_metrics_.initial_vertex_balance << "                             │\n";
        std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << max_vertex_ratio << "                             │\n";
        std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << vertex_balance_improvement << "%                             │\n";
        std::cout << "│                                                             │\n";
        std::cout << "│ Edge Balance:                                               │\n";
        std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << phase1_metrics_.initial_edge_balance << "                             │\n";
        std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << max_edge_ratio << "                             │\n";
        std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_balance_improvement << "%                             │\n";
        std::cout << "│                                                             │\n";
        std::cout << "│ 실행 시간:                                                  │\n";
        std::cout << "│   Phase 1 (로딩): " << std::setw(6) << phase1_metrics_.loading_time_ms << " ms                          │\n";
        std::cout << "│   Phase 1 (분산): " << std::setw(6) << phase1_metrics_.distribution_time_ms << " ms                          │\n";
        std::cout << "│   Phase 2 (7단계): " << std::setw(5) << execution_time_ms << " ms                          │\n";
        std::cout << "│   총 소요시간:     " << std::setw(5) << (phase1_metrics_.loading_time_ms + phase1_metrics_.distribution_time_ms + execution_time_ms) << " ms                          │\n";
        std::cout << "└─────────────────────────────────────────────────────────────┘\n";
        
        // 개선 요약
        std::cout << "\n=== 알고리즘 성능 요약 ===\n";
        if (edge_cut_improvement > 0) {
            std::cout << "✓ Edge-cut " << std::fixed << std::setprecision(1) << edge_cut_improvement << "% 개선 (" 
                      << phase1_metrics_.initial_edge_cut << " → " << current_edge_cut_ << ")\n";
        } else {
            std::cout << "⚠ Edge-cut " << std::fixed << std::setprecision(1) << -edge_cut_improvement << "% 악화 (" 
                      << phase1_metrics_.initial_edge_cut << " → " << current_edge_cut_ << ")\n";
        }
        
        if (vertex_balance_improvement > 0) {
            std::cout << "✓ Vertex Balance " << std::fixed << std::setprecision(1) << vertex_balance_improvement << "% 개선\n";
        } else {
            std::cout << "⚠ Vertex Balance " << std::fixed << std::setprecision(1) << -vertex_balance_improvement << "% 악화\n";
        }
        
        if (edge_balance_improvement > 0) {
            std::cout << "✓ Edge Balance " << std::fixed << std::setprecision(1) << edge_balance_improvement << "% 개선\n";
        } else {
            std::cout << "⚠ Edge Balance " << std::fixed << std::setprecision(1) << -edge_balance_improvement << "% 악화\n";
        }
        
        std::cout << "\n=== 7단계 알고리즘 완료 ===\n";
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
