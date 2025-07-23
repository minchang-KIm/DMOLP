#include "mpi_workflow.h"

// Step 5: 진짜 고스트 노드 교환 (선택적 점대점 통신)
void MPIDistributedWorkflowV2::exchangePartitionUpdates() {
    std::cout << "Step5 Rank " << mpi_rank_ << ": 진짜 고스트 노드 교환 (선택적 통신)\n";
    
    // === 1. 고스트 노드 요청 생성 ===
    // 1-1. 내가 필요한 고스트 정점들 식별 (이웃 프로세스의 정점들)
    std::unordered_map<int, std::vector<int>> ghost_requests; // {target_rank: [vertex_ids]}
    
    for (int boundary_vertex : BV_) {
        // 경계 정점의 모든 이웃을 확인
        for (int edge_idx = local_graph_.row_ptr[boundary_vertex]; 
             edge_idx < local_graph_.row_ptr[boundary_vertex + 1]; ++edge_idx) {
            int neighbor = local_graph_.col_indices[edge_idx];
            
            // 다른 프로세스 소유 정점인 경우
            if (neighbor < start_vertex_ || neighbor >= end_vertex_) {
                int target_rank = getOwnerRank(neighbor);
                if (target_rank != mpi_rank_ && target_rank >= 0 && target_rank < mpi_size_) {
                    ghost_requests[target_rank].push_back(neighbor);
                }
            }
        }
    }
    
    // 중복 제거
    for (auto& [rank, vertices] : ghost_requests) {
        std::sort(vertices.begin(), vertices.end());
        vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
    }
    
    std::cout << "  고스트 요청: " << ghost_requests.size() << "개 프로세스에게 요청\n";
    
    // === 2. 비동기 점대점 통신으로 고스트 요청/응답 ===
    std::vector<MPI_Request> send_requests, recv_requests;
    std::vector<std::vector<int>> send_buffers, recv_buffers;
    
    // 2-1. 고스트 요청 전송
    for (auto& [target_rank, vertices] : ghost_requests) {
        if (!vertices.empty()) {
            send_buffers.emplace_back();
            auto& buffer = send_buffers.back();
            
            // 버퍼 구성: [count, vertex1, vertex2, ...]
            buffer.push_back(static_cast<int>(vertices.size()));
            buffer.insert(buffer.end(), vertices.begin(), vertices.end());
            
            MPI_Request req;
            MPI_Isend(buffer.data(), buffer.size(), MPI_INT, target_rank, 
                     100, MPI_COMM_WORLD, &req);
            send_requests.push_back(req);
        }
    }
    
    // 2-2. 고스트 요청 수신 및 응답
    recv_buffers.resize(mpi_size_);
    for (int source_rank = 0; source_rank < mpi_size_; ++source_rank) {
        if (source_rank != mpi_rank_) {
            // 먼저 요청 크기 수신
            int request_size;
            MPI_Status status;
            MPI_Probe(source_rank, 100, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &request_size);
            
            if (request_size > 0) {
                recv_buffers[source_rank].resize(request_size);
                MPI_Recv(recv_buffers[source_rank].data(), request_size, MPI_INT,
                        source_rank, 100, MPI_COMM_WORLD, &status);
                
                // 요청 처리 및 응답 준비
                processGhostRequest(source_rank, recv_buffers[source_rank]);
            }
        }
    }
    
    // 2-3. 전송 완료 대기
    if (!send_requests.empty()) {
        MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
    }
    
    std::cout << "  고스트 노드 교환 완료\n";
    
    // === 3. PU_RO 처리 (새로운 경계 정점) ===
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
    
    std::cout << "교환완료 Rank " << mpi_rank_ << ": " << new_boundary_candidates.size() 
              << "개 새 경계 정점 추가\n";
}

// 고스트 요청 처리 및 응답
void MPIDistributedWorkflowV2::processGhostRequest(int source_rank, const std::vector<int>& request_buffer) {
    if (request_buffer.empty()) return;
    
    int count = request_buffer[0];
    std::vector<int> response_buffer;
    response_buffer.push_back(count); // 응답 개수
    
    // 요청된 정점들의 라벨 정보 수집
    for (int i = 1; i <= count && i < request_buffer.size(); ++i) {
        int vertex_id = request_buffer[i];
        
        // 내가 소유한 정점인지 확인
        if (vertex_id >= start_vertex_ && vertex_id < end_vertex_) {
            int local_index = vertex_id - start_vertex_;
            if (local_index >= 0 && local_index < vertex_labels_.size()) {
                response_buffer.push_back(vertex_id);
                response_buffer.push_back(vertex_labels_[local_index]);
            }
        }
    }
    
    // 비동기 응답 전송
    if (response_buffer.size() > 1) {
        MPI_Request req;
        MPI_Isend(response_buffer.data(), response_buffer.size(), MPI_INT,
                 source_rank, 200, MPI_COMM_WORLD, &req);
        // 요청 완료는 나중에 일괄 처리
    }
}

// 정점 소유자 랭크 계산
int MPIDistributedWorkflowV2::getOwnerRank(int vertex_id) const {
    // 정점 ID 범위로 소유자 결정
    int vertices_per_rank = (local_graph_.num_vertices + mpi_size_ - 1) / mpi_size_;
    int owner = vertex_id / vertices_per_rank;
    return (owner >= 0 && owner < mpi_size_) ? owner : -1;
}

// Step 6: 수렴 확인 (안전성 개선)
bool MPIDistributedWorkflowV2::checkConvergence() {
    std::cout << "Step6 Rank " << mpi_rank_ << ": 수렴 확인\n";
    
    // 안전한 변수 초기화 확인
    if (convergence_count_ < 0) {
        convergence_count_ = 0; // 음수값 방지
    }
    
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
void MPIDistributedWorkflowV2::prepareNextIteration() {
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

// Ghost Node 정보 교환 (MPI 통신) - 대용량 데이터 처리 개선
void MPIDistributedWorkflowV2::exchangeGhostNodeInformation(std::vector<std::pair<int, int>>& ghost_updates) {
    std::cout << "  [Ghost Node] MPI 프로세스 간 Ghost Node 정보 교환 중...\n";
    
    // 큰 데이터 처리를 위한 크기 제한
    const int MAX_GHOST_REQUESTS = 100000; // 최대 요청 수 제한
    
    // 각 MPI 프로세스가 소유하지 않는 정점들의 라벨 정보를 수집
    std::vector<int> remote_vertices;  // 다른 프로세스 소유 정점들
    std::vector<int> remote_labels;    // 해당 정점들의 최신 라벨
    
    // 경계 정점들의 이웃 중 다른 프로세스 소유 정점 찾기 (제한된 수만)
    for (int boundary_vertex : BV_) {
        if (remote_vertices.size() >= MAX_GHOST_REQUESTS) break; // 제한 초과 시 중단
        
        for (int edge_idx = local_graph_.row_ptr[boundary_vertex]; 
             edge_idx < local_graph_.row_ptr[boundary_vertex + 1]; ++edge_idx) {
            if (remote_vertices.size() >= MAX_GHOST_REQUESTS) break;
            
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
    
    // 크기 제한 적용
    if (remote_vertices.size() > MAX_GHOST_REQUESTS) {
        remote_vertices.resize(MAX_GHOST_REQUESTS);
        std::cout << "  [Ghost Node] Warning: 요청 수를 " << MAX_GHOST_REQUESTS << "개로 제한\n";
    }
    
    std::cout << "  [Ghost Node] " << remote_vertices.size() << "개 원격 정점의 Ghost Node 정보 요청\n";
    
    // === MPI 통신으로 Ghost Node 정보 교환 ===
    // 1. 각 프로세스의 요청 개수 수집
    int num_requests = remote_vertices.size();
    std::vector<int> request_counts(mpi_size_);
    MPI_Allgather(&num_requests, 1, MPI_INT, request_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // 2. 전체 요청 크기 및 오프셋 계산
    std::vector<int> request_displs(mpi_size_);
    int total_requests = 0;
    for (int i = 0; i < mpi_size_; ++i) {
        request_displs[i] = total_requests;
        total_requests += request_counts[i];
    }
    
    // 3. 모든 프로세스의 요청 정점 ID들 수집
    std::vector<int> all_requests(total_requests);
    MPI_Allgatherv(remote_vertices.data(), num_requests, MPI_INT,
                   all_requests.data(), request_counts.data(), request_displs.data(), 
                   MPI_INT, MPI_COMM_WORLD);
    
    // 4. 각 프로세스가 소유한 정점들에 대해서만 라벨 응답 준비
    std::vector<int> my_responses;
    std::vector<int> my_vertex_ids;
    
    for (int vertex_id : all_requests) {
        if (vertex_id >= start_vertex_ && vertex_id < end_vertex_) {
            int local_index = vertex_id - start_vertex_;
            if (local_index >= 0 && local_index < vertex_labels_.size()) {
                my_vertex_ids.push_back(vertex_id);
                my_responses.push_back(vertex_labels_[local_index]);
            }
        }
    }
    
    // 5. 각 프로세스의 응답 개수 수집
    int my_response_count = my_responses.size();
    std::vector<int> response_counts(mpi_size_);
    MPI_Allgather(&my_response_count, 1, MPI_INT, response_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // 6. 응답 오프셋 계산
    std::vector<int> response_displs(mpi_size_);
    int total_responses = 0;
    for (int i = 0; i < mpi_size_; ++i) {
        response_displs[i] = total_responses;
        total_responses += response_counts[i];
    }
    
    // 7. 정점 ID와 라벨을 함께 교환 (직렬화)
    std::vector<int> my_serialized_responses;
    for (int i = 0; i < my_vertex_ids.size(); ++i) {
        my_serialized_responses.push_back(my_vertex_ids[i]);  // vertex_id
        my_serialized_responses.push_back(my_responses[i]);   // label
    }
    
    std::vector<int> all_serialized_responses(total_responses * 2);
    std::vector<int> serialized_counts(mpi_size_);
    std::vector<int> serialized_displs(mpi_size_);
    
    for (int i = 0; i < mpi_size_; ++i) {
        serialized_counts[i] = response_counts[i] * 2;  // vertex_id + label
        serialized_displs[i] = response_displs[i] * 2;
    }
    
    MPI_Allgatherv(my_serialized_responses.data(), my_serialized_responses.size(), MPI_INT,
                   all_serialized_responses.data(), serialized_counts.data(), serialized_displs.data(),
                   MPI_INT, MPI_COMM_WORLD);
    
    // 8. Ghost Node 업데이트 정보 구성
    std::unordered_map<int, int> received_labels;
    for (int i = 0; i < all_serialized_responses.size(); i += 2) {
        int vertex_id = all_serialized_responses[i];
        int label = all_serialized_responses[i + 1];
        received_labels[vertex_id] = label;
    }
    
    // 9. 요청했던 정점들의 라벨 정보만 추출
    for (int vertex_id : remote_vertices) {
        if (received_labels.find(vertex_id) != received_labels.end()) {
            ghost_updates.push_back({vertex_id, received_labels[vertex_id]});
        }
    }
    
    std::cout << "  [Ghost Node] " << ghost_updates.size() << "개 Ghost Node 업데이트 수신 완료\n";
}

// Ghost Node 업데이트 적용
void MPIDistributedWorkflowV2::applyGhostNodeUpdates(const std::vector<std::pair<int, int>>& ghost_updates) {
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
void MPIDistributedWorkflowV2::updatePartitionBoundaries() {
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
