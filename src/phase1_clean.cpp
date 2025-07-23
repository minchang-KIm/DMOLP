#ifndef PHASE1_CU
#define PHASE1_CU

#include "phase1.h"
#include <set>
#include <unordered_map>

// 실제 그래프 파일 로딩 함수
bool loadGraphFromFile(const std::string& filename, Graph& graph, std::vector<int>& vertex_labels, int num_partitions) {
    std::string full_path;
    
    // 파일 경로 확인
    if (filename.find('/') == std::string::npos) {
        // 상대 경로인 경우 여러 위치에서 찾기
        std::vector<std::string> search_paths = {
            "/home/intern_graph/intern/" + filename,
            "/home/intern_graph/intern/datasets/" + filename,
            "/home/intern_graph/build/" + filename,
            "./" + filename
        };
        
        bool found = false;
        for (const auto& path : search_paths) {
            std::ifstream test_file(path);
            if (test_file.good()) {
                full_path = path;
                found = true;
                break;
            }
        }
        
        if (!found) {
            std::cerr << "파일을 찾을 수 없습니다: " << filename << std::endl;
            return false;
        }
    } else {
        full_path = filename;
    }
    
    std::cout << "그래프 파일 로딩: " << full_path << std::endl;
    
    std::ifstream file(full_path);
    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다: " << full_path << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line);
    std::istringstream header(line);
    
    // 첫 번째 줄에서 최대 정점 ID 읽기
    int max_vertex_id;
    header >> max_vertex_id;
    
    std::cout << "그래프 정보: 최대 정점 ID " << max_vertex_id << " (sparse adjacency list format)\n";
    
    // 실제 사용되는 정점들과 그 이웃들을 저장
    std::unordered_map<int, std::vector<int>> sparse_adj_list;
    std::set<int> unique_vertices;
    
    // adjacency list 형식으로 읽기 - 각 줄의 첫 번째는 정점 ID, 나머지는 이웃들
    int line_count = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        
        int vertex_id;
        if (!(iss >> vertex_id)) continue; // 빈 줄이나 잘못된 형식 건너뛰기
        
        unique_vertices.insert(vertex_id);
        
        int neighbor;
        while (iss >> neighbor) {
            if (neighbor >= 0 && neighbor != vertex_id) {
                sparse_adj_list[vertex_id].push_back(neighbor);
                unique_vertices.insert(neighbor);
                // 무방향 그래프이므로 양방향 추가 (나중에 중복 제거)
                sparse_adj_list[neighbor].push_back(vertex_id);
            }
        }
        
        line_count++;
        if (line_count % 100000 == 0) {
            std::cout << "처리된 라인: " << line_count << ", 고유 정점: " << unique_vertices.size() << std::endl;
        }
    }
    
    // 실제 정점 수 계산
    graph.num_vertices = unique_vertices.size();
    std::cout << "파싱 완료: " << graph.num_vertices << "개 정점, " << line_count << "개 라인 처리\n";
    
    // 정점 ID를 0부터 시작하는 연속 ID로 매핑
    std::unordered_map<int, int> vertex_id_map;
    std::vector<int> reverse_map(graph.num_vertices);
    int mapped_id = 0;
    for (int original_id : unique_vertices) {
        vertex_id_map[original_id] = mapped_id;
        reverse_map[mapped_id] = original_id;
        mapped_id++;
    }
    
    // 매핑된 ID로 인접 리스트 생성
    std::vector<std::vector<int>> adj_list(graph.num_vertices);
    for (const auto& pair : sparse_adj_list) {
        int original_u = pair.first;
        if (vertex_id_map.find(original_u) == vertex_id_map.end()) continue;
        
        int mapped_u = vertex_id_map[original_u];
        for (int original_v : pair.second) {
            if (vertex_id_map.find(original_v) != vertex_id_map.end()) {
                int mapped_v = vertex_id_map[original_v];
                adj_list[mapped_u].push_back(mapped_v);
            }
        }
    }
    
    // 중복 간선 제거 및 정렬
    int total_edges = 0;
    for (int u = 0; u < graph.num_vertices; ++u) {
        std::sort(adj_list[u].begin(), adj_list[u].end());
        adj_list[u].erase(std::unique(adj_list[u].begin(), adj_list[u].end()), adj_list[u].end());
        total_edges += adj_list[u].size();
    }
    
    std::cout << "인접 리스트 생성 완료: " << total_edges << "개 간선 (방향성)\n";
    
    // CSR 형식으로 변환
    graph.row_ptr.resize(graph.num_vertices + 1, 0);
    graph.col_indices.resize(total_edges);
    vertex_labels.resize(graph.num_vertices);
    
    int edge_count = 0;
    for (int u = 0; u < graph.num_vertices; ++u) {
        graph.row_ptr[u] = edge_count;
        for (int v : adj_list[u]) {
            graph.col_indices[edge_count++] = v;
        }
    }
    graph.row_ptr[graph.num_vertices] = edge_count;
    graph.num_edges = edge_count;
    
    // 초기 라벨 할당 (랜덤)
    for (int i = 0; i < graph.num_vertices; ++i) {
        vertex_labels[i] = i % num_partitions;
    }
    
    file.close();
    std::cout << "그래프 로딩 완료: " << graph.num_vertices << "개 정점, " << graph.num_edges << "개 간선\n";
    return true;
}

// MPI를 통한 그래프 분산 (대용량 데이터 처리 개선)
Graph distributeGraphViaMPI(const Graph& global_graph, const std::vector<int>& global_vertex_labels,
                           std::vector<int>& local_vertex_labels, int mpi_rank, int mpi_size) {
    Graph local_graph;
    
    // 단순한 정점 분할 (균등 분배)
    int vertices_per_rank = global_graph.num_vertices / mpi_size;
    int start_vertex = mpi_rank * vertices_per_rank;
    int end_vertex = (mpi_rank == mpi_size - 1) ? global_graph.num_vertices : (mpi_rank + 1) * vertices_per_rank;
    
    local_graph.num_vertices = end_vertex - start_vertex;
    
    // 안전한 범위 체크
    if (start_vertex >= 0 && end_vertex <= global_vertex_labels.size()) {
        local_vertex_labels.assign(global_vertex_labels.begin() + start_vertex, 
                                   global_vertex_labels.begin() + end_vertex);
    } else {
        std::cerr << "Error: Invalid vertex range for rank " << mpi_rank << std::endl;
        local_vertex_labels.resize(local_graph.num_vertices, 0);
    }
    
    // 로컬 그래프의 CSR 구성 (메모리 효율적)
    local_graph.row_ptr.resize(local_graph.num_vertices + 1, 0);
    std::vector<int> temp_edges;
    temp_edges.reserve(local_graph.num_vertices * 10); // 예상 간선 수로 예약
    
    int edge_count = 0;
    for (int u = 0; u < local_graph.num_vertices; ++u) {
        int global_u = start_vertex + u;
        local_graph.row_ptr[u] = edge_count;
        
        // 안전한 범위 체크
        if (global_u >= 0 && global_u < global_graph.num_vertices) {
            for (int edge_idx = global_graph.row_ptr[global_u]; 
                 edge_idx < global_graph.row_ptr[global_u + 1]; ++edge_idx) {
                if (edge_idx < global_graph.col_indices.size()) {
                    int global_v = global_graph.col_indices[edge_idx];
                    temp_edges.push_back(global_v);
                    edge_count++;
                }
            }
        }
    }
    local_graph.row_ptr[local_graph.num_vertices] = edge_count;
    
    local_graph.col_indices = std::move(temp_edges);
    local_graph.num_edges = edge_count;
    
    return local_graph;
}

// Phase 1 메트릭 계산
Phase1Metrics calculatePhase1Metrics(const Graph& global_graph, const std::vector<int>& vertex_labels,
                                    int num_partitions, long loading_time_ms, long distribution_time_ms) {
    Phase1Metrics metrics;
    metrics.loading_time_ms = loading_time_ms;
    metrics.distribution_time_ms = distribution_time_ms;
    metrics.total_vertices = global_graph.num_vertices;
    metrics.total_edges = global_graph.num_edges;
    
    // 파티션별 정점/간선 수 계산을 위한 임시 배열
    std::vector<int> partition_vertex_counts(num_partitions, 0);
    std::vector<int> partition_edge_counts(num_partitions, 0);
    
    // 파티션별 정점 수 계산
    for (int i = 0; i < global_graph.num_vertices; ++i) {
        int label = vertex_labels[i];
        if (label >= 0 && label < num_partitions) {
            partition_vertex_counts[label]++;
        }
    }
    
    // 파티션별 간선 수 계산 (내부 간선만)
    for (int u = 0; u < global_graph.num_vertices; ++u) {
        int u_label = vertex_labels[u];
        if (u_label < 0 || u_label >= num_partitions) continue;
        
        for (int edge_idx = global_graph.row_ptr[u]; edge_idx < global_graph.row_ptr[u + 1]; ++edge_idx) {
            int v = global_graph.col_indices[edge_idx];
            if (v < global_graph.num_vertices) {
                int v_label = vertex_labels[v];
                if (u_label == v_label) {
                    partition_edge_counts[u_label]++;
                }
            }
        }
    }
    
    // Edge-cut 계산
    int edge_cut = 0;
    for (int u = 0; u < global_graph.num_vertices; ++u) {
        int u_label = vertex_labels[u];
        for (int edge_idx = global_graph.row_ptr[u]; edge_idx < global_graph.row_ptr[u + 1]; ++edge_idx) {
            int v = global_graph.col_indices[edge_idx];
            if (v < global_graph.num_vertices && u < v) { // 중복 방지
                int v_label = vertex_labels[v];
                if (u_label != v_label) {
                    edge_cut++;
                }
            }
        }
    }
    metrics.initial_edge_cut = edge_cut;
    
    // Vertex Balance 계산
    double max_vertex_ratio = 0.0;
    double expected_vertices_per_partition = static_cast<double>(global_graph.num_vertices) / num_partitions;
    for (int i = 0; i < num_partitions; ++i) {
        double ratio = static_cast<double>(partition_vertex_counts[i]) / expected_vertices_per_partition;
        max_vertex_ratio = std::max(max_vertex_ratio, ratio);
    }
    metrics.initial_vertex_balance = max_vertex_ratio;
    
    // Edge Balance 계산
    int total_partition_edges = 0;
    for (int i = 0; i < num_partitions; ++i) {
        total_partition_edges += partition_edge_counts[i];
    }
    
    // 디버그: 파티션별 정점/간선 수 출력
    std::cout << "=== Phase 1 파티션별 상세 정보 ===\n";
    for (int i = 0; i < num_partitions; ++i) {
        std::cout << "파티션 " << i << ": " << partition_vertex_counts[i] 
                  << "개 정점, " << partition_edge_counts[i] << "개 내부 간선\n";
    }
    std::cout << "총 내부 간선: " << total_partition_edges << " / 전체 간선: " << global_graph.num_edges << "\n";
    
    double max_edge_ratio = 0.0;
    if (total_partition_edges > 0) {
        double expected_edges_per_partition = static_cast<double>(total_partition_edges) / num_partitions;
        std::cout << "예상 파티션별 간선 수: " << expected_edges_per_partition << "\n";
        for (int i = 0; i < num_partitions; ++i) {
            double ratio = static_cast<double>(partition_edge_counts[i]) / expected_edges_per_partition;
            std::cout << "파티션 " << i << " 간선 비율: " << ratio << "\n";
            max_edge_ratio = std::max(max_edge_ratio, ratio);
        }
    }
    metrics.initial_edge_balance = max_edge_ratio;
    
    return metrics;
}

// Phase1 메인 함수
Phase1Metrics phase1_partition_and_distribute(int mpi_rank, int mpi_size, int num_partitions,
                                             Graph& local_graph, std::vector<int>& vertex_labels,
                                             const std::string& filename) {
    auto start_loading = std::chrono::high_resolution_clock::now();
    
    Graph global_graph;
    std::vector<int> global_vertex_labels;
    Phase1Metrics global_metrics;
    
    // 마스터 프로세스에서 그래프 로딩
    if (mpi_rank == 0) {
        std::cout << "\n=== Phase 1: 그래프 분할 및 분배 ===\n";
        if (!loadGraphFromFile(filename, global_graph, global_vertex_labels, num_partitions)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        auto end_loading = std::chrono::high_resolution_clock::now();
        auto loading_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_loading - start_loading);
        
        auto start_distribution = std::chrono::high_resolution_clock::now();
        
        // Phase 1 메트릭 계산
        global_metrics = calculatePhase1Metrics(global_graph, global_vertex_labels, num_partitions, 
                                               loading_time.count(), 0);
        
        std::cout << "\n=== Phase 1 초기 메트릭 ===\n";
        std::cout << "Edge-cut: " << global_metrics.initial_edge_cut << "\n";
        std::cout << "Vertex Balance: " << global_metrics.initial_vertex_balance << "\n";
        std::cout << "Edge Balance: " << global_metrics.initial_edge_balance << "\n";
        std::cout << "로딩 시간: " << global_metrics.loading_time_ms << " ms\n";
        
        auto end_distribution = std::chrono::high_resolution_clock::now();
        auto distribution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_distribution - start_distribution);
        global_metrics.distribution_time_ms = distribution_time.count();
    }
    
    // 메트릭 정보 브로드캐스트
    MPI_Bcast(&global_metrics, sizeof(Phase1Metrics), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // 그래프 기본 정보 브로드캐스트
    int graph_info[2] = {0, 0}; // {num_vertices, num_edges}
    if (mpi_rank == 0) {
        graph_info[0] = global_graph.num_vertices;
        graph_info[1] = global_graph.num_edges;
    }
    MPI_Bcast(graph_info, 2, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 모든 프로세스에서 그래프 분산 실행
    if (mpi_rank == 0) {
        local_graph = distributeGraphViaMPI(global_graph, global_vertex_labels, vertex_labels, mpi_rank, mpi_size);
        
        // 다른 프로세스들에게 그래프 데이터 전송
        for (int rank = 1; rank < mpi_size; ++rank) {
            // 각 프로세스의 정점 범위 계산
            int vertices_per_rank = global_graph.num_vertices / mpi_size;
            int start_vertex = rank * vertices_per_rank;
            int end_vertex = (rank == mpi_size - 1) ? global_graph.num_vertices : (rank + 1) * vertices_per_rank;
            int local_num_vertices = end_vertex - start_vertex;
            
            // 기본 정보 전송
            int local_info[3] = {local_num_vertices, start_vertex, end_vertex};
            MPI_Send(local_info, 3, MPI_INT, rank, 0, MPI_COMM_WORLD);
            
            // 정점 라벨 전송
            if (local_num_vertices > 0 && start_vertex + local_num_vertices <= global_vertex_labels.size()) {
                MPI_Send(global_vertex_labels.data() + start_vertex, local_num_vertices, MPI_INT, rank, 1, MPI_COMM_WORLD);
            } else {
                std::cerr << "Rank 0 경고: 정점 라벨 전송 범위 초과 (rank " << rank << ")" << std::endl;
                // 빈 데이터 전송
                std::vector<int> empty_labels(local_num_vertices, 0);
                MPI_Send(empty_labels.data(), local_num_vertices, MPI_INT, rank, 1, MPI_COMM_WORLD);
            }
            
            // CSR 데이터 전송 (안전성 체크)
            std::vector<int> local_row_ptr(local_num_vertices + 1, 0);
            std::vector<int> local_col_indices;
            
            int edge_count = 0;
            for (int u = 0; u < local_num_vertices; ++u) {
                int global_u = start_vertex + u;
                local_row_ptr[u] = edge_count;
                
                if (global_u >= 0 && global_u < global_graph.num_vertices && 
                    global_u < global_graph.row_ptr.size() - 1) {
                    for (int edge_idx = global_graph.row_ptr[global_u]; 
                         edge_idx < global_graph.row_ptr[global_u + 1] && 
                         edge_idx < global_graph.col_indices.size(); ++edge_idx) {
                        local_col_indices.push_back(global_graph.col_indices[edge_idx]);
                        edge_count++;
                    }
                }
            }
            local_row_ptr[local_num_vertices] = edge_count;
            
            // row_ptr 전송
            MPI_Send(local_row_ptr.data(), local_num_vertices + 1, MPI_INT, rank, 2, MPI_COMM_WORLD);
            
            // col_indices 전송
            if (edge_count > 0) {
                MPI_Send(local_col_indices.data(), edge_count, MPI_INT, rank, 3, MPI_COMM_WORLD);
            }
        }
    } else {
        // 비마스터 프로세스: 데이터 수신
        int local_info[3];
        MPI_Recv(local_info, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        local_graph.num_vertices = local_info[0];
        int start_vertex = local_info[1];
        int end_vertex = local_info[2];
        
        std::cout << "Rank " << mpi_rank << " 수신 시작: " << local_graph.num_vertices 
                  << "개 정점 예상" << std::endl;
        
        // 정점 라벨 수신
        if (local_graph.num_vertices > 0) {
            vertex_labels.resize(local_graph.num_vertices);
            MPI_Recv(vertex_labels.data(), local_graph.num_vertices, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Rank " << mpi_rank << " 라벨 수신 완료: " << vertex_labels.size() << "개" << std::endl;
        }
        
        // CSR 데이터 수신
        local_graph.row_ptr.resize(local_graph.num_vertices + 1);
        MPI_Recv(local_graph.row_ptr.data(), local_graph.num_vertices + 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        local_graph.num_edges = local_graph.row_ptr[local_graph.num_vertices];
        std::cout << "Rank " << mpi_rank << " row_ptr 수신 완료: " << local_graph.num_edges << "개 간선" << std::endl;
        
        if (local_graph.num_edges > 0) {
            local_graph.col_indices.resize(local_graph.num_edges);
            MPI_Recv(local_graph.col_indices.data(), local_graph.num_edges, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Rank " << mpi_rank << " col_indices 수신 완료" << std::endl;
        }
        
        std::cout << "Rank " << mpi_rank << " 수신 완료: " << local_graph.num_vertices 
                  << "개 정점, " << local_graph.num_edges << "개 간선\n";
    }
    
    if (mpi_rank == 0) {
        std::cout << "\n=== Phase 1 완료 ===\n";
        std::cout << "총 소요시간: " << (global_metrics.loading_time_ms + global_metrics.distribution_time_ms) << " ms\n";
    }
    
    return global_metrics;
}

#endif // PHASE1_CU
