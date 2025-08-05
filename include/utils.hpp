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

struct NodeInfo {
    int vertex;
    std::vector<int> neighbors;
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
};

struct PartitionUpdate {
    int partition_id;
    int node;

    PartitionUpdate() : partition_id(-1), node(-1) {}
    PartitionUpdate(int pid, int num) : partition_id(pid), node(num) {}
};

void sync_vector(int procId, int sourceProc, std::vector<int> &vec);
std::vector<int> serialize_node_info(const std::vector<NodeInfo> &nodes);
std::vector<NodeInfo> deserialize_node_info(const std::vector<int> &buffer);
std::vector<int> serialize_updates(const std::vector<PartitionUpdate> &updates);
std::vector<PartitionUpdate> deserialize_updates(const std::vector<int> &buffer);
void sync_updates(int procId, int nprocs, const std::vector<PartitionUpdate> &local_updates, std::vector<std::vector<int>> &partitions, std::unordered_set<int> &global_partitioned);
std::vector<int> serialize_partitions(const std::vector<std::vector<int>> &partitions);
std::vector<std::vector<int>> deserialize_partitions(const std::vector<int> &buffer);
void sync_global_partitions(int procId, int nprocs, std::vector<std::vector<int>> &partitions, std::unordered_set<int> &global_partitioned, std::vector<PartitionUpdate> &pending_updates);
void sync_partitioned_status(int procId, int nprocs, std::unordered_set<int> &global_partitioned);
void sync_partitioned(int procId, int nprocs, const std::vector<int> &newly_partitioned, std::unordered_set<int> &global_partitioned);
void add_node_to_partition(int node, int partition_id, std::vector<std::vector<int>> &partitions, std::unordered_set<int> &global_partitioned, std::vector<PartitionUpdate> &pending_updates);
void seed_redistribution(int procId, int nprocs, int numParts, const std::vector<int> &remaining_seeds, const std::unordered_map<int, std::vector<int>> &local_adj, std::vector<std::vector<int>> &partitions, std::unordered_set<int> &global_partitioned);
void print_summary(int procId, int nprocs, const std::vector<std::vector<int>> &partitions, const std::unordered_map<int, int> &global_degree);
void print_proc_partition(int procId, int nprocs, int numParts, const std::vector<std::vector<int>> &partitions, const std::unordered_map<int, std::vector<int>> &local_adj);
void printComparisonReport(const PartitioningMetrics& m1, const PartitioningMetrics& m2);

#endif