#ifndef UTILS_HPP
#define UTILS_HPP

#pragma once

#include "phase1/phase1.h"
#include "graph_types.h"
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <roaring/roaring.h>

bool is_local_node(int node, int procId, int nprocs);
roaring_bitmap_t* create_partition_bitmap(const std::vector<int> &partition_nodes);
std::unordered_map<int, roaring_bitmap_t*> convert_adj(const std::unordered_map<int, std::vector<int>> &local_adj);
void free_converted_graph(std::unordered_map<int, roaring_bitmap_t*> &bitmap_map);
roaring_bitmap_t* create_hub_bitmap(const std::vector<int> &hub_nodes);
void broadcast_roaring_bitmap(roaring_bitmap_t *bitmap, int root, MPI_Comm comm);
void allreduce_roaring_bitmap_or(roaring_bitmap_t *local_bitmap, roaring_bitmap_t *result_bitmap, MPI_Comm comm);
void sync_vector(int procId, int sourceProc, std::vector<int> &vec);
std::vector<int> serialize_updates(const std::vector<PartitionUpdate> &updates);
std::vector<PartitionUpdate> deserialize_updates(const std::vector<int> &buffer);
std::vector<PartitionUpdate> collect_updates(int procId, int nprocs, const std::vector<PartitionUpdate> &local_updates);
void apply_updates(const std::vector<PartitionUpdate> &updates, std::vector<std::vector<int>> &partitions);
void sync_updates(int procId, int nprocs, const std::vector<PartitionUpdate> &updates, std::vector<std::vector<int>> &partitions);
void sync_partitioned_status(int procId, int nprocs, const std::vector<PartitionUpdate> &updates, std::unordered_set<int> &global_partitioned);
void print_summary(int procId, int nprocs, const std::vector<std::vector<int>> &partitions, const std::unordered_map<int, int> &global_degree);
void printComparisonReport(const PartitioningMetrics& m1, const PartitioningMetrics& m2);

#endif