#ifndef DMOLP_PHASE1_H
#define DMOLP_PHASE1_H

#include "types.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

// Phase1 관련 함수 선언
bool loadGraphFromFile(const std::string& filename, Graph& graph, 
                      std::vector<int>& vertex_labels, int num_partitions);

Graph distributeGraphViaMPI(const Graph& global_graph, 
                           const std::vector<int>& global_vertex_labels,
                           std::vector<int>& local_vertex_labels,
                           int mpi_rank, int mpi_size);

Phase1Metrics calculatePhase1Metrics(const Graph& global_graph, 
                                    const std::vector<int>& vertex_labels,
                                    int num_partitions, 
                                    long loading_time_ms, 
                                    long distribution_time_ms);

// Phase1 메인 함수
Phase1Metrics phase1_partition_and_distribute(int mpi_rank, int mpi_size, 
                                             int num_partitions,
                                             Graph& local_graph, 
                                             std::vector<int>& vertex_labels,
                                             const std::string& filename);

#endif // DMOLP_PHASE1_H
