#ifndef SEED_HPP
#define SEED_HPP

#pragma once
#include <vector>
#include <unordered_map>

std::vector<int> find_seeds(const int procId, const int nprocs, const int numParts, const size_t num_nodes, const std::pair<int, int> &first_seed, const std::vector<int> &hub_nodes, const std::unordered_map<int, int> &global_degree, const std::unordered_map<int, std::vector<int>> &adj);

#endif