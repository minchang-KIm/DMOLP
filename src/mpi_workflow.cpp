#include "mpi_workflow.h"
#include "phase1.h"

#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif

// 생성자
MPIDistributedWorkflowV2::MPIDistributedWorkflowV2(int argc, char* argv[], const Graph& phase1_graph, 
                                                   const std::vector<int>& phase1_labels, 
                                                   const Phase1Metrics& phase1_metrics) {
    // MPI는 이미 메인에서 초기화됨
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    
    // 변수 안전 초기화
    current_edge_cut_ = 0;
    previous_edge_cut_ = 0;
    edge_rate_ = 0.0;
    convergence_count_ = 0;  // 명시적으로 0으로 초기화
    
    num_partitions_ = std::atoi(argv[2]);
    
    // 파라미터 검증
    if (num_partitions_ <= 0) {
        std::cerr << "Rank " << mpi_rank_ << " 오류: 잘못된 파티션 수 " << num_partitions_ << std::endl;
        num_partitions_ = 2; // 기본값으로 설정
    }
    
    // Phase 1에서 전달받은 그래프와 라벨 사용
    local_graph_ = phase1_graph;
    vertex_labels_ = phase1_labels;
    phase1_metrics_ = phase1_metrics;
    
    // 디버깅: 데이터 크기 확인 및 검증
    std::cout << "Rank " << mpi_rank_ << " 생성자 데이터:" << std::endl;
    std::cout << "  vertex_labels_.size() = " << vertex_labels_.size() << std::endl;
    std::cout << "  local_graph_.num_vertices = " << local_graph_.num_vertices << std::endl;
    std::cout << "  local_graph_.num_edges = " << local_graph_.num_edges << std::endl;
    std::cout << "  phase1_metrics.total_vertices = " << phase1_metrics.total_vertices << std::endl;
    std::cout.flush(); // 강제 출력
    
    // 데이터 일관성 체크
    if (vertex_labels_.size() != local_graph_.num_vertices) {
        std::cerr << "Rank " << mpi_rank_ << " 경고: 라벨 크기와 정점 수 불일치!" << std::endl;
        std::cerr << "  예상: " << local_graph_.num_vertices << ", 실제: " << vertex_labels_.size() << std::endl;
        // 안전을 위해 라벨 배열 크기 조정
        vertex_labels_.resize(local_graph_.num_vertices, 0);
        std::cerr << "  크기 조정 완료: " << vertex_labels_.size() << std::endl;
    }
    
    // 정점 소유권 계산 (Phase 1과 동일한 로직)
    int vertices_per_rank = phase1_metrics.total_vertices / mpi_size_;
    start_vertex_ = mpi_rank_ * vertices_per_rank;
    end_vertex_ = (mpi_rank_ == mpi_size_ - 1) ? phase1_metrics.total_vertices : (mpi_rank_ + 1) * vertices_per_rank;
    
    std::cout << "  start_vertex_ = " << start_vertex_ << ", end_vertex_ = " << end_vertex_ << std::endl;
    std::cout.flush(); // 강제 출력
    
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

// 소멸자
MPIDistributedWorkflowV2::~MPIDistributedWorkflowV2() {
    // 안전한 메모리 해제
    vertex_labels_.clear();
    BV_.clear();
    PU_.PU_OV.clear();
    PU_.PU_ON.clear();
    PU_.PU_RV.clear();
    PU_.PU_RN.clear();
    PU_.PU_RO.clear();
    PI_.clear();
    NV_.clear();
    
    // 그래프 데이터 해제
    local_graph_.row_ptr.clear();
    local_graph_.col_indices.clear();
    
#ifdef USE_CUDA
    // GPU 메모리 매니저들 안전하게 해제
    for (auto& manager : gpu_managers_) {
        if (manager) {
            manager.reset(); // unique_ptr 안전 해제
        }
    }
    gpu_managers_.clear();
    gpu_properties_.clear();
    
    // GPU 컨텍스트 정리
    if (num_gpus_ > 0) {
        cudaDeviceReset(); // 모든 GPU 컨텍스트 정리
    }
#endif
    
    // MPI_Finalize은 메인에서 처리
}

// 메인 실행 함수
void MPIDistributedWorkflowV2::run() {
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
