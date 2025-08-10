#ifndef PARTITION_HPP
#define PARTITION_HPP

#pragma once
#include <vector>
#include <unordered_map>

#include "graph_types.h" 

void partition_expansion(int procId, int nprocs, int numParts, const std::vector<int> &seeds, const std::unordered_map<int, int> &global_degree, const std::unordered_map<int, std::vector<int>> &local_adj, std::unordered_map<int, Graph> &local_partition_graphs, std::unordered_map<int, GhostNodes> &local_partition_ghosts);

#endif