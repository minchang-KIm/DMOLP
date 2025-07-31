#pragma once
#include <vector>
#include <string>

#include "graph_types.h"

bool loadGraphFromFile(const std::string& filename, Graph& graph, std::vector<int>& vertex_labels, int num_partitions);
Phase1Metrics calculatePhase1Metrics(const Graph& graph, const std::vector<int>& vertex_labels, int num_partitions, long loading_time, long distribution_time);
Phase1Metrics phase1_partition_and_distribute(int mpi_rank, int mpi_size, int num_partitions, Graph& local_graph, std::vector<int>& vertex_labels, const std::string& filename);
