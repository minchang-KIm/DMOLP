#include "mpi_workflow.h"

#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif

// Step 4: Dynamic Unweighted LP 수행 (핵심!)
void MPIDistributedWorkflowV2::performDynamicLabelPropagation() {
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
// CUDA 상수 메모리에 파티션 정보 복사 함수
void MPIDistributedWorkflowV2::updatePartitionInfoOnGPU() {
    if (PI_.size() > 16) {
        std::cerr << "Error: 파티션 수가 16개를 초과했습니다!" << std::endl;
        return;
    }
    
#ifdef USE_CUDA
    // TODO: GPU 파티션 정보 업데이트는 GPUMemoryManager에서 처리
    std::cout << "  [GPU] 파티션 정보 업데이트 (GPUMemoryManager로 위임)\n";
#endif
}

// Multi-GPU 구현 (스레드 풀 + GPU 큐 방식)
int MPIDistributedWorkflowV2::performDynamicLabelPropagationGPU() {
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

#else
// CPU 모드에서는 빈 함수들
void MPIDistributedWorkflowV2::updatePartitionInfoOnGPU() {
    // CPU 모드에서는 아무것도 하지 않음
}

// CUDA 비활성화 시 더미 함수
int MPIDistributedWorkflowV2::performDynamicLabelPropagationGPU() {
    return performDynamicLabelPropagationCPU();
}
#endif

// CPU OpenMP 구현 (올바른 로직)
int MPIDistributedWorkflowV2::performDynamicLabelPropagationCPU() {
#ifdef USE_CUDA
    std::cout << "  [GPU] CUDA 병렬 처리 시작\n";
#else
    std::cout << "  [CPU] OpenMP 병렬 처리 시작\n";
#endif
    
    if (BV_.empty()) {
#ifdef USE_CUDA
        std::cout << "  [GPU] 경계 정점 없음, 라벨 변경 없음\n";
#else
        std::cout << "  [CPU] 경계 정점 없음, 라벨 변경 없음\n";
#endif
        return 0;
    }
    
#ifdef USE_CUDA
    std::cout << "  [GPU] " << BV_.size() << "개 경계 정점에서 Label Propagation 수행\n";
#else
    std::cout << "  [CPU] " << BV_.size() << "개 경계 정점에서 Label Propagation 수행\n";
#endif
    
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
    
#ifdef USE_CUDA
    std::cout << "  [GPU] " << updates_count << "개 라벨 변경 완료\n";
#else
    std::cout << "  [CPU] " << updates_count << "개 라벨 변경 완료\n";
#endif
    return updates_count;
}
