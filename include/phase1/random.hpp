#ifndef RANDOM_HPP
#define RANDOM_HPP

#pragma once
#include <vector>
#include <unordered_map>

#include "graph_types.h" 

void random_partition(
    int procId, int nprocs, 
    int numParts, 
    int theta,
    const std::vector<int> &seeds, 
    const std::unordered_map<int, int> &global_degree, 
    const std::unordered_map<int, std::vector<int>> &local_adj,
    Graph &local_partition_graphs, 
    GhostNodes &local_partition_ghosts,
    bool verbose
);

#endif