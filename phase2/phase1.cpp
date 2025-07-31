#include "phase1.h"
#include "graph_types.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>


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
            filename
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
            std::cerr << "그래프 파일을 찾을 수 없습니다: " << filename << std::endl;
            return false;
        }
    } else {
        full_path = filename;
    }
    
    std::ifstream file(full_path);
    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다: " << full_path << std::endl;
        return false;
    }
    
    std::cout << "그래프 파일 로딩: " << full_path << std::endl;
    
    std::string line;
    bool is_metis_format = false;
    
    // 첫 번째 줄 읽기
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int> first_line_nums;
        int num;
        while (iss >> num) {
            first_line_nums.push_back(num);
        }
        
        // METIS 형태 감지 (첫 줄에 정점 수, 간선 수가 있음)
        if (first_line_nums.size() == 2 && first_line_nums[0] > 0 && first_line_nums[1] > 0) {
            is_metis_format = true;
            graph.num_vertices = first_line_nums[0];
            int expected_edges = first_line_nums[1] * 2; // METIS는 undirected이므로 각 간선이 2번 카운트됨
            
            std::cout << "METIS 형태 감지: " << graph.num_vertices << "개 정점, " << first_line_nums[1] << "개 간선" << std::endl;
            
            // METIS 형태 파싱
            std::vector<std::vector<int>> adj_list(graph.num_vertices);
            int vertex_id = 0;
            
            while (std::getline(file, line) && vertex_id < graph.num_vertices) {
                if (line.empty()) continue;
                
                std::istringstream iss(line);
                int neighbor;
                while (iss >> neighbor) {
                    if (neighbor > 0 && neighbor <= graph.num_vertices) {
                        adj_list[vertex_id].push_back(neighbor - 1); // METIS는 1-indexed
                    }
                }
                vertex_id++;
            }
            
            // CSR 형태로 변환
            graph.num_edges = 0;
            for (const auto& neighbors : adj_list) {
                graph.num_edges += neighbors.size();
            }
            
            graph.row_ptr.resize(graph.num_vertices + 1);
            graph.col_indices.resize(graph.num_edges);
            
            graph.row_ptr[0] = 0;
            int edge_idx = 0;
            
            for (int i = 0; i < graph.num_vertices; ++i) {
                for (int neighbor : adj_list[i]) {
                    graph.col_indices[edge_idx++] = neighbor;
                }
                graph.row_ptr[i + 1] = edge_idx;
            }
        } else {
            // ADJ 형태로 가정하고 파싱
            file.seekg(0); // 파일 처음으로 돌아가기
            
            std::vector<std::vector<int>> adj_list;
            int max_vertex = 0;
            
            while (std::getline(file, line)) {
                if (line.empty() || line[0] == '#') continue;
                
                std::istringstream iss(line);
                std::vector<int> neighbors;
                int vertex;
                
                while (iss >> vertex) {
                    neighbors.push_back(vertex);
                    max_vertex = std::max(max_vertex, vertex);
                }
                
                if (!neighbors.empty()) {
                    adj_list.push_back(neighbors);
                }
            }
            
            graph.num_vertices = adj_list.size();
            graph.num_edges = 0;
            
            for (const auto& neighbors : adj_list) {
                graph.num_edges += neighbors.size();
            }
            
            if (graph.num_vertices == 0) {
                std::cerr << "유효한 그래프 데이터가 없습니다." << std::endl;
                return false;
            }
            
            // CSR 구성
            graph.row_ptr.resize(graph.num_vertices + 1);
            graph.col_indices.resize(graph.num_edges);
            
            graph.row_ptr[0] = 0;
            int edge_idx = 0;
            
            for (int i = 0; i < graph.num_vertices; ++i) {
                for (int neighbor : adj_list[i]) {
                    if (neighbor < graph.num_vertices) {
                        graph.col_indices[edge_idx++] = neighbor;
                    }
                }
                graph.row_ptr[i + 1] = edge_idx;
            }
            
            graph.num_edges = edge_idx;
            graph.col_indices.resize(graph.num_edges);
        }
    }
    
    // 초기 라벨 할당 (더 나은 분산을 위해 해시 기반)
    vertex_labels.resize(graph.num_vertices);
    for (int i = 0; i < graph.num_vertices; ++i) {
        vertex_labels[i] = (i * 31 + 17) % num_partitions; // 간단한 해시 함수
    }
    
    std::cout << "그래프 로딩 완료: " << graph.num_vertices << "개 정점, " << graph.num_edges << "개 간선" << std::endl;
    return true;
}

// Phase 1 초기 메트릭 계산
Phase1Metrics calculatePhase1Metrics(const Graph& graph, const std::vector<int>& vertex_labels, int num_partitions, long loading_time, long distribution_time) {
    Phase1Metrics metrics;
    metrics.loading_time_ms = loading_time;
    metrics.distribution_time_ms = distribution_time;
    metrics.total_vertices = graph.num_vertices;
    metrics.total_edges = graph.num_edges;
    metrics.partition_vertex_counts.resize(num_partitions, 0);
    metrics.partition_edge_counts.resize(num_partitions, 0);
    
    // 파티션별 정점 수 계산
    for (int i = 0; i < graph.num_vertices; ++i) {
        int label = vertex_labels[i];
        if (label >= 0 && label < num_partitions) {
            metrics.partition_vertex_counts[label]++;
        }
    }
    
    // 파티션별 간선 수 및 초기 edge-cut 계산
    metrics.initial_edge_cut = 0;
    for (int u = 0; u < graph.num_vertices; ++u) {
        int u_label = vertex_labels[u];
        if (u_label < 0 || u_label >= num_partitions) continue;
        
        for (int edge_idx = graph.row_ptr[u]; edge_idx < graph.row_ptr[u + 1]; ++edge_idx) {
            int v = graph.col_indices[edge_idx];
            if (v < graph.num_vertices) {
                int v_label = vertex_labels[v];
                if (u_label == v_label) {
                    metrics.partition_edge_counts[u_label]++;
                } else if (u < v) { // 중복 방지
                    metrics.initial_edge_cut++;
                }
            }
        }
    }
    
    // 초기 vertex balance 계산: VB = max_i |V_i| / (|V|/k)
    double max_vertex_ratio = 0.0;
    double expected_vertices_per_partition = static_cast<double>(graph.num_vertices) / num_partitions;
    for (int i = 0; i < num_partitions; ++i) {
        double ratio = static_cast<double>(metrics.partition_vertex_counts[i]) / expected_vertices_per_partition;
        max_vertex_ratio = std::max(max_vertex_ratio, ratio);
    }
    metrics.initial_vertex_balance = max_vertex_ratio;
    
    // 초기 edge balance 계산: EB = max_i |E_i| / (|E|/k)
    double max_edge_ratio = 0.0;
    int total_partition_edges = 0;
    for (int i = 0; i < num_partitions; ++i) {
        total_partition_edges += metrics.partition_edge_counts[i];
    }
    if (total_partition_edges > 0) {
        double expected_edges_per_partition = static_cast<double>(total_partition_edges) / num_partitions;
        for (int i = 0; i < num_partitions; ++i) {
            double ratio = static_cast<double>(metrics.partition_edge_counts[i]) / expected_edges_per_partition;
            max_edge_ratio = std::max(max_edge_ratio, ratio);
        }
    }
    metrics.initial_edge_balance = max_edge_ratio;
    
    return metrics;
}

// Phase 1: 그래프 입력 받아 라벨(파티션)별로 분할 후 각 서버에 분배
Phase1Metrics phase1_partition_and_distribute(int mpi_rank, int mpi_size, int num_partitions, Graph& local_graph, std::vector<int>& vertex_labels, const std::string& filename) {
    auto phase1_start = std::chrono::high_resolution_clock::now();
    
    Graph full_graph;
    std::vector<int> full_labels;
    Phase1Metrics global_metrics;
    
    // Rank 0에서 전체 그래프 로딩
    if (mpi_rank == 0) {
        auto load_start = std::chrono::high_resolution_clock::now();
        
        if (!loadGraphFromFile(filename, full_graph, full_labels, num_partitions)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return global_metrics;
        }
        
        auto load_end = std::chrono::high_resolution_clock::now();
        long loading_time = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
        
        // Phase 1 초기 메트릭 계산
        global_metrics = calculatePhase1Metrics(full_graph, full_labels, num_partitions, loading_time, 0);
        
        std::cout << "\n=== Phase 1 초기 메트릭 ===\n";
        std::cout << "그래프 로딩 시간: " << loading_time << " ms\n";
        std::cout << "초기 Edge-cut: " << global_metrics.initial_edge_cut << "\n";
        std::cout << "초기 Vertex Balance: " << global_metrics.initial_vertex_balance << "\n";
        std::cout << "초기 Edge Balance: " << global_metrics.initial_edge_balance << "\n";
        std::cout << "파티션별 정점 분포:\n";
        for (int i = 0; i < num_partitions; ++i) {
            std::cout << "  파티션 " << i << ": " << global_metrics.partition_vertex_counts[i] << "개 정점, " 
                      << global_metrics.partition_edge_counts[i] << "개 간선\n";
        }
    }
    
    auto distribution_start = std::chrono::high_resolution_clock::now();
    
    // 그래프 크기 정보 브로드캐스트
    int graph_info[2] = {0, 0};
    if (mpi_rank == 0) {
        graph_info[0] = full_graph.num_vertices;
        graph_info[1] = full_graph.num_edges;
    }
    MPI_Bcast(graph_info, 2, MPI_INT, 0, MPI_COMM_WORLD);
    
    int total_vertices = graph_info[0];
    int total_edges = graph_info[1];
    
    // 각 MPI 프로세스가 담당할 정점 범위 계산
    int vertices_per_rank = total_vertices / mpi_size;
    int start_vertex = mpi_rank * vertices_per_rank;
    int end_vertex = (mpi_rank == mpi_size - 1) ? total_vertices : (mpi_rank + 1) * vertices_per_rank;
    
    local_graph.num_vertices = end_vertex - start_vertex;
    
    if (mpi_rank == 0) {
        // Rank 0: 자신의 부분 복사
        local_graph.num_edges = full_graph.row_ptr[end_vertex] - full_graph.row_ptr[start_vertex];
        local_graph.row_ptr.resize(local_graph.num_vertices + 1);
        local_graph.col_indices.resize(local_graph.num_edges);
        vertex_labels.resize(local_graph.num_vertices);
        
        // 데이터 복사
        for (int i = 0; i <= local_graph.num_vertices; ++i) {
            local_graph.row_ptr[i] = full_graph.row_ptr[start_vertex + i] - full_graph.row_ptr[start_vertex];
        }
        
        for (int i = 0; i < local_graph.num_edges; ++i) {
            local_graph.col_indices[i] = full_graph.col_indices[full_graph.row_ptr[start_vertex] + i];
        }
        
        for (int i = 0; i < local_graph.num_vertices; ++i) {
            vertex_labels[i] = full_labels[start_vertex + i];
        }
        
        // 다른 프로세스들에게 데이터 전송
        for (int rank = 1; rank < mpi_size; ++rank) {
            int rank_start = rank * vertices_per_rank;
            int rank_end = (rank == mpi_size - 1) ? total_vertices : (rank + 1) * vertices_per_rank;
            int rank_vertices = rank_end - rank_start;
            int rank_edges = full_graph.row_ptr[rank_end] - full_graph.row_ptr[rank_start];
            
            // 크기 정보 전송
            int sizes[2] = {rank_vertices, rank_edges};
            MPI_Send(sizes, 2, MPI_INT, rank, 0, MPI_COMM_WORLD);
            
            // row_ptr 전송
            std::vector<int> rank_row_ptr(rank_vertices + 1);
            for (int i = 0; i <= rank_vertices; ++i) {
                rank_row_ptr[i] = full_graph.row_ptr[rank_start + i] - full_graph.row_ptr[rank_start];
            }
            MPI_Send(rank_row_ptr.data(), rank_vertices + 1, MPI_INT, rank, 1, MPI_COMM_WORLD);
            
            // col_indices 전송
            MPI_Send(&full_graph.col_indices[full_graph.row_ptr[rank_start]], rank_edges, MPI_INT, rank, 2, MPI_COMM_WORLD);
            
            // vertex_labels 전송
            MPI_Send(&full_labels[rank_start], rank_vertices, MPI_INT, rank, 3, MPI_COMM_WORLD);
        }
    } else {
        // 다른 프로세스들: 데이터 수신
        int sizes[2];
        MPI_Recv(sizes, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        local_graph.num_vertices = sizes[0];
        local_graph.num_edges = sizes[1];
        
        local_graph.row_ptr.resize(local_graph.num_vertices + 1);
        local_graph.col_indices.resize(local_graph.num_edges);
        vertex_labels.resize(local_graph.num_vertices);
        
        MPI_Recv(local_graph.row_ptr.data(), local_graph.num_vertices + 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_graph.col_indices.data(), local_graph.num_edges, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(vertex_labels.data(), local_graph.num_vertices, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    auto distribution_end = std::chrono::high_resolution_clock::now();
    long distribution_time = std::chrono::duration_cast<std::chrono::milliseconds>(distribution_end - distribution_start).count();
    
    if (mpi_rank == 0) {
        global_metrics.distribution_time_ms = distribution_time;
        std::cout << "MPI 분산 시간: " << distribution_time << " ms\n";
    }
    
    // 파티션 담당 범위 계산
    int partitions_per_rank = num_partitions / mpi_size;
    int start_partition = mpi_rank * partitions_per_rank;
    int end_partition = (mpi_rank == mpi_size-1) ? num_partitions : (mpi_rank+1)*partitions_per_rank;
    
    std::cout << "[Phase1] Rank " << mpi_rank << ": " << local_graph.num_vertices << "개 정점, "
              << local_graph.num_edges << "개 간선, 파티션 " << start_partition << "~" << (end_partition-1) << " 담당 (정점 " 
              << start_vertex << "~" << (end_vertex-1) << ")\n";
    
    // 전체 시간 계산
    auto phase1_end = std::chrono::high_resolution_clock::now();
    long total_phase1_time = std::chrono::duration_cast<std::chrono::milliseconds>(phase1_end - phase1_start).count();
    
    if (mpi_rank == 0) {
        std::cout << "Phase 1 총 소요시간: " << total_phase1_time << " ms\n";
        std::cout << "=== Phase 1 완료 ===\n";
        
        global_metrics.distribution_time_ms = distribution_time;
    }
    return global_metrics;
}
