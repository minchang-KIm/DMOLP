#pragma once
#include <vector>
#include <string>

struct Graph {
    int num_vertices;
    int num_edges;
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
    std::vector<int> vertex_ids;
};

struct Phase1Metrics {
    int initial_edge_cut;
    double initial_vertex_balance;
    double initial_edge_balance;
    long loading_time_ms;
    long distribution_time_ms;
    int total_vertices;
    int total_edges;
    std::vector<int> partition_vertex_counts;
    std::vector<int> partition_edge_counts;
};

bool loadGraphFromFile(const std::string& filename, Graph& graph, std::vector<int>& vertex_labels, int num_partitions);
Phase1Metrics calculatePhase1Metrics(const Graph& graph, const std::vector<int>& vertex_labels, int num_partitions, long loading_time, long distribution_time);
Phase1Metrics phase1_partition_and_distribute(int mpi_rank, int mpi_size, int num_partitions, Graph& local_graph, std::vector<int>& vertex_labels, const std::string& filename);
