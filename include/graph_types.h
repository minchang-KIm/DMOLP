#ifndef DMOLP_GRAPH_TYPES_H
#define DMOLP_GRAPH_TYPES_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <boost/dynamic_bitset.hpp>

struct BFSResult {
    std::vector<boost::dynamic_bitset<>> levels;
    boost::dynamic_bitset<> all_visited;

    BFSResult(size_t num_nodes = 0) : all_visited(num_nodes) {}

    void clear() {
        levels.clear();
        all_visited.clear();
    }

    void ensure_level(int level, size_t num_nodes) {
        if (level >= static_cast<int>(levels.size())) levels.resize(level + 1, boost::dynamic_bitset<>(num_nodes));
    }

    void print() const {
        std::cout << "=== BFS Result ===" << std::endl;
        std::cout << "Number of levels: " << levels.size() << std::endl;
        std::cout << "All visited nodes count: " << all_visited.count() << std::endl;
        
        for (size_t i = 0; i < levels.size(); ++i) {
            std::cout << "Level " << i << " nodes count: " << levels[i].count() << std::endl;
            if (levels[i].count() > 0) {
                std::cout << "  Nodes: ";
                for (size_t j = 0; j < levels[i].size(); ++j) {
                    if (levels[i][j]) {
                        std::cout << j << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }
};

struct Delta {
    int gid;        // global id
    int new_label;  // new label

    void print() const {
        std::cout << "=== Delta ===" << std::endl;
        std::cout << "Global ID: " << gid << std::endl;
        std::cout << "New Label: " << new_label << std::endl;
        std::cout << std::endl;
    }
};

struct NodeInfo {
    int vertex;
    std::vector<int> neighbors;

    void print() const {
        std::cout << "=== Node Info ===" << std::endl;
        std::cout << "Vertex: " << vertex << std::endl;
        std::cout << "Neighbors count: " << neighbors.size() << std::endl;
        std::cout << "Neighbors: ";
        for (int neighbor : neighbors) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl << std::endl;
    }
};

struct FrontierNode {
    int vertex;
    int partition_id;
    double ratio;
    int partition_degree;
    int total_degree;

    FrontierNode() : vertex(-1), partition_id(-1), ratio(0.0), partition_degree(0), total_degree(0) {}
    FrontierNode(int v, double r, int pd, int td, int pid = -1) : vertex(v), partition_id(pid), ratio(r), partition_degree(pd), total_degree(td) {}

    bool operator<(const FrontierNode &other) const {
        return ratio < other.ratio;
    }

    bool operator>(const FrontierNode &other) const {
        return ratio > other.ratio;
    }

    void print() const {
        std::cout << "=== Frontier Node ===" << std::endl;
        std::cout << "Vertex: " << vertex << std::endl;
        std::cout << "Partition ID: " << partition_id << std::endl;
        std::cout << "Ratio: " << std::fixed << std::setprecision(4) << ratio << std::endl;
        std::cout << "Partition Degree: " << partition_degree << std::endl;
        std::cout << "Total Degree: " << total_degree << std::endl;
        std::cout << std::endl;
    }
};

struct PartitionUpdate {
    int partition_id;
    int node;

    PartitionUpdate() : partition_id(-1), node(-1) {}
    PartitionUpdate(int pid, int num) : partition_id(pid), node(num) {}

    void print() const {
        std::cout << "=== Partition Update ===" << std::endl;
        std::cout << "Partition ID: " << partition_id << std::endl;
        std::cout << "Node: " << node << std::endl;
        std::cout << std::endl;
    }
};

// 그래프 구조체 (CSR) 각 프로세스마다 로컬 그래프를 표현
// global_ids: 로컬 노드의 글로벌 ID 배열
// vertex_labels: 로컬 노드의 라벨 (소유 노드만 변경 가능)
// row_ptr: 각 정점의 시작 인덱스 (CSR 형식)
// col_indices: 인접 정점의 인덱스 배열 (로컬: 0~num_vertices-1, Ghost: num_vertices~)
// num_edges: 총 엣지 수
// num_vertices: 로컬 정점 수
struct Graph {
    int num_vertices = 0;
    int num_edges = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
    std::vector<int> global_ids; // 로컬 노드의 글로벌 ID 배열
    std::vector<int> vertex_labels; // 로컬 노드의 라벨

    void clear() {
        num_vertices = 0;
        num_edges = 0;
        row_ptr.clear();
        col_indices.clear();
        global_ids.clear();
        vertex_labels.clear();
    }

    void print() const {
        std::cout << "=== Graph ===" << std::endl;
        std::cout << "Number of vertices: " << num_vertices << std::endl;
        std::cout << "Number of edges: " << num_edges << std::endl;
        
        std::cout << "Global IDs: ";
        for (int gid : global_ids) {
            std::cout << gid << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Vertex Labels: ";
        for (int label : vertex_labels) {
            std::cout << label << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Row pointers: ";
        for (int ptr : row_ptr) {
            std::cout << ptr << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Column indices: ";
        for (int idx : col_indices) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
        
        // 인접 리스트 형태로 출력
        std::cout << "Adjacency list representation:" << std::endl;
        for (int i = 0; i < num_vertices; ++i) {
            if (i < static_cast<int>(global_ids.size()) && i < static_cast<int>(vertex_labels.size())) {
                std::cout << "Vertex " << i << " (global_id=" << global_ids[i] 
                          << ", label=" << vertex_labels[i] << "): ";
                if (i + 1 < static_cast<int>(row_ptr.size())) {
                    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                        if (j < static_cast<int>(col_indices.size())) {
                            std::cout << col_indices[j] << " ";
                        }
                    }
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }
};

// Ghost 노드 정보 구조체 각 스레드마다 Ghost 노드 정보를 표현
// global_ids: Ghost 노드의 글로벌 ID 배열
// ghost_labels: Ghost 노드의 라벨 배열
// global_to_local: 글로벌 ID → 로컬 인덱스 매핑
struct GhostNodes {
    std::vector<int> global_ids;  // 글로벌 ID
    std::vector<int> ghost_labels; // Ghost 노드의 라벨
    std::unordered_map<int, int> global_to_local; // 글로벌 → 로컬 인덱스 매핑

    void clear() {
        global_ids.clear();
        ghost_labels.clear();
        global_to_local.clear();
    }

    void print() const {
        std::cout << "=== Ghost Nodes ===" << std::endl;
        std::cout << "Number of ghost nodes: " << global_ids.size() << std::endl;
        
        std::cout << "Global IDs: ";
        for (int gid : global_ids) {
            std::cout << gid << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Ghost Labels: ";
        for (int label : ghost_labels) {
            std::cout << label << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Global to Local mapping:" << std::endl;
        for (const auto& pair : global_to_local) {
            std::cout << "  Global " << pair.first << " -> Local " << pair.second << std::endl;
        }
        std::cout << std::endl;
    }
};

// 파티션별 통계 구조체
struct PartitionStats {
    std::vector<int> local_vertex_counts;
    std::vector<int> local_edge_counts;
    std::vector<int> global_vertex_counts;
    std::vector<int> global_edge_counts;
    int total_vertices;
    uint64_t total_edges;
    double expected_vertices;
    double expected_edges;

    void print() const {
        std::cout << "=== Partition Stats ===" << std::endl;
        std::cout << "Total vertices: " << total_vertices << std::endl;
        std::cout << "Total edges: " << total_edges << std::endl;
        std::cout << "Expected vertices: " << std::fixed << std::setprecision(2) << expected_vertices << std::endl;
        std::cout << "Expected edges: " << std::fixed << std::setprecision(2) << expected_edges << std::endl;
        
        std::cout << "Local vertex counts: ";
        for (int count : local_vertex_counts) {
            std::cout << count << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Local edge counts: ";
        for (int count : local_edge_counts) {
            std::cout << count << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Global vertex counts: ";
        for (int count : global_vertex_counts) {
            std::cout << count << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Global edge counts: ";
        for (int count : global_edge_counts) {
            std::cout << count << " ";
        }
        std::cout << std::endl << std::endl;
    }
};

// Phase 1 메트릭 구조체 
struct Phase1Metrics {
    int initial_edge_cut = 0;
    double initial_vertex_balance = 0.0;
    double initial_edge_balance = 0.0;
    long loading_time_ms = 0;
    long partition_time_ms = 0;
    long distribution_time_ms = 0;
    int total_vertices = 0;
    uint64_t total_edges = 0;
    std::vector<int> partition_vertex_counts;
    std::vector<int> partition_edge_counts;

    void print() const {
        std::cout << "=== Phase 1 Metrics ===" << std::endl;
        std::cout << "Initial edge cut: " << initial_edge_cut << std::endl;
        std::cout << "Initial vertex balance: " << std::fixed << std::setprecision(4) << initial_vertex_balance << std::endl;
        std::cout << "Initial edge balance: " << std::fixed << std::setprecision(4) << initial_edge_balance << std::endl;
        std::cout << "Loading time: " << loading_time_ms << " ms" << std::endl;
        std::cout << "Partitioning time: " << partition_time_ms << " ms" << std::endl;
        std::cout << "Distribution time: " << distribution_time_ms << " ms" << std::endl;
        std::cout << "Total vertices: " << total_vertices << std::endl;
        std::cout << "Total edges: " << total_edges << std::endl;
        
        std::cout << "Partition vertex counts: ";
        for (int count : partition_vertex_counts) {
            std::cout << count << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Partition edge counts: ";
        for (int count : partition_edge_counts) {
            std::cout << count << " ";
        }
        std::cout << std::endl << std::endl;
    }
};

// Phase 1/2 공통 비교용 메트릭 구조체
struct PartitioningMetrics {
    int edge_cut = 0;
    double vertex_balance = 0.0;
    double edge_balance = 0.0;
    long loading_time_ms = 0;
    long distribution_time_ms = 0;
    int num_partitions = 0;
    int total_vertices = 0;
    uint64_t total_edges = 0;

    PartitioningMetrics() = default;
    PartitioningMetrics(const Phase1Metrics& m, int num_parts) {
        edge_cut = m.initial_edge_cut;
        vertex_balance = m.initial_vertex_balance;
        edge_balance = m.initial_edge_balance;
        loading_time_ms = m.loading_time_ms;
        distribution_time_ms = m.distribution_time_ms;
        num_partitions = num_parts;
        total_vertices = m.total_vertices;
        total_edges = m.total_edges;
    }

    void print() const {
        std::cout << "=== Partitioning Metrics ===" << std::endl;
        std::cout << "Edge cut: " << edge_cut << std::endl;
        std::cout << "Vertex balance: " << std::fixed << std::setprecision(4) << vertex_balance << std::endl;
        std::cout << "Edge balance: " << std::fixed << std::setprecision(4) << edge_balance << std::endl;
        std::cout << "Loading time: " << loading_time_ms << " ms" << std::endl;
        std::cout << "Distribution time: " << distribution_time_ms << " ms" << std::endl;
        std::cout << "Number of partitions: " << num_partitions << std::endl;
        std::cout << "Total vertices: " << total_vertices << std::endl;
        std::cout << "Total edges: " << total_edges << std::endl;
        std::cout << std::endl;
    }
};
#endif // DMOLP_GRAPH_TYPES_H
