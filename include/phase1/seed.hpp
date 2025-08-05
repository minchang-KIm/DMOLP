#ifndef SEED_HPP
#define SEED_HPP

#pragma once
#include <vector>
#include <unordered_map>

std::vector<int> find_seeds(const int procId, const int nprocs, const int numParts, const std::vector<int> &landmarks, const std::vector<int> &hub_nodes, const std::unordered_map<int, std::vector<int>> &adj);

#endif