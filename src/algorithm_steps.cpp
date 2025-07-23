#include "mpi_workflow.h"

#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif

// Step 1: RV, RE 계산 (안전성 개선)
void MPIDistributedWorkflowV2::calculateRatios() {
    std::cout << "Step1 Rank " << mpi_rank_ << ": RV, RE 계산\n";
    
    // 안전성 체크
    if (vertex_labels_.size() < local_graph_.num_vertices) {
        std::cerr << "Error Rank " << mpi_rank_ << ": vertex_labels_ 크기 부족 (" 
                  << vertex_labels_.size() << " < " << local_graph_.num_vertices << ")\n";
        return;
    }
    
    // 각 파티션별 정점/간선 수 계산
    std::vector<int> vertex_counts(num_partitions_, 0);
    std::vector<int> edge_counts(num_partitions_, 0);
    
    // 정점 수 계산 (안전한 범위 체크)
    for (int i = 0; i < local_graph_.num_vertices && i < vertex_labels_.size(); ++i) {
        int label = vertex_labels_[i];
        if (label >= 0 && label < num_partitions_) {
            vertex_counts[label]++;
        }
    }
    
    // 간선 수 계산 (실제 CSR 사용, 안전한 범위 체크)
    for (int u = 0; u < local_graph_.num_vertices && u < vertex_labels_.size(); ++u) {
        int u_label = vertex_labels_[u];
        if (u_label < 0 || u_label >= num_partitions_) continue;
        
        if (u < local_graph_.row_ptr.size() - 1) {
            for (int edge_idx = local_graph_.row_ptr[u]; 
                 edge_idx < local_graph_.row_ptr[u + 1] && edge_idx < local_graph_.col_indices.size(); 
                 ++edge_idx) {
                int v = local_graph_.col_indices[edge_idx];
                
                // 로컬 정점인지 확인
                if (v >= start_vertex_ && v < end_vertex_) {
                    int local_v = v - start_vertex_;
                    if (local_v >= 0 && local_v < vertex_labels_.size()) {
                        int v_label = vertex_labels_[local_v];
                        if (v_label == u_label) {
                            edge_counts[u_label]++;
                        }
                    }
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
void MPIDistributedWorkflowV2::calculateImbalance() {
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
void MPIDistributedWorkflowV2::calculateEdgeCutAndExtractBoundary() {
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
void MPIDistributedWorkflowV2::extractBoundaryVerticesWithOpenMP() {
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
void MPIDistributedWorkflowV2::extractBoundaryVertices() {
    extractBoundaryVerticesWithOpenMP();
}
