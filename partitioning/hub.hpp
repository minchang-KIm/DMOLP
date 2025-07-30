#pragma once
#include <vector>
#include <unordered_map>

std::vector<int> find_hub_nodes(const std::unordered_map<int, int> &global_degree);
std::vector<int> find_landmarks(const std::unordered_map<int, int> &global_degree);