#pragma once
#include <vector>
#include <unordered_map>

void partition_expansion(int procId, int nprocs, int numParts, int theta, const std::vector<int> &seeds, const std::unordered_map<int, int> &global_degree, const std::unordered_map<int, std::vector<int>> &local_adj, std::vector<std::vector<int>> &partitions);