#ifndef PHASE1_CU
#define PHASE1_CU

#include "phase1.h"

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
    
    header >> graph.num_vertices >> graph.num_edges;
    
    std::cout << "그래프 정보: " << graph.num_vertices << "개 정점, " << graph.num_edges << "개 간선\n";
    
    // CSR 형식으로 변환
    graph.row_ptr.resize(graph.num_vertices + 1, 0);
    graph.col_indices.resize(graph.num_edges);
    vertex_labels.resize(graph.num_vertices);
    
    // 임시 인접 리스트
    std::vector<std::vector<int>> adj_list(graph.num_vertices);
    
    // 간선 읽기
    for (int i = 0; i < graph.num_edges; ++i) {
        int u, v;
        file >> u >> v;
        adj_list[u].push_back(v);
        if (u != v) { // 자기 루프 방지
            adj_list[v].push_back(u);
        }
    }
    
    // CSR 형식으로 변환
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

// MPI를 통한 그래프 분산
Graph distributeGraphViaMPI(const Graph& global_graph, const std::vector<int>& global_vertex_labels,
                           std::vector<int>& local_vertex_labels, int mpi_rank, int mpi_size) {
    Graph local_graph;
    
    // 단순한 정점 분할 (균등 분배)
    int vertices_per_rank = global_graph.num_vertices / mpi_size;
    int start_vertex = mpi_rank * vertices_per_rank;
    int end_vertex = (mpi_rank == mpi_size - 1) ? global_graph.num_vertices : (mpi_rank + 1) * vertices_per_rank;
    
    local_graph.num_vertices = end_vertex - start_vertex;
    local_vertex_labels.assign(global_vertex_labels.begin() + start_vertex, 
                               global_vertex_labels.begin() + end_vertex);
    
    // 로컬 그래프의 CSR 구성
    local_graph.row_ptr.resize(local_graph.num_vertices + 1);
    std::vector<int> temp_edges;
    
    int edge_count = 0;
    for (int u = 0; u < local_graph.num_vertices; ++u) {
        int global_u = start_vertex + u;
        local_graph.row_ptr[u] = edge_count;
        
        for (int edge_idx = global_graph.row_ptr[global_u]; 
             edge_idx < global_graph.row_ptr[global_u + 1]; ++edge_idx) {
            int global_v = global_graph.col_indices[edge_idx];
            temp_edges.push_back(global_v);
            edge_count++;
        }
    }
    local_graph.row_ptr[local_graph.num_vertices] = edge_count;
    
    local_graph.col_indices = temp_edges;
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
    double max_edge_ratio = 0.0;
    if (total_partition_edges > 0) {
        double expected_edges_per_partition = static_cast<double>(total_partition_edges) / num_partitions;
        for (int i = 0; i < num_partitions; ++i) {
            double ratio = static_cast<double>(partition_edge_counts[i]) / expected_edges_per_partition;
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
    
    // 그래프 분산
    if (mpi_rank == 0) {
        local_graph = distributeGraphViaMPI(global_graph, global_vertex_labels, vertex_labels, mpi_rank, mpi_size);
    } else {
        // 비마스터 프로세스는 빈 그래프로 시작
        local_graph.num_vertices = 0;
        local_graph.num_edges = 0;
        vertex_labels.clear();
    }
    
    if (mpi_rank == 0) {
        std::cout << "\n=== Phase 1 완료 ===\n";
        std::cout << "총 소요시간: " << (global_metrics.loading_time_ms + global_metrics.distribution_time_ms) << " ms\n";
    }
    
    return global_metrics;
}

#endif // PHASE1_CU
