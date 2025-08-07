#ifndef INIT_HPP
#define INIT_HPP

#pragma once
#include <vector>
#include <unordered_map>

void load_graph(const char *filename, int procId, int nprocs, std::unordered_map<int, std::vector<int>> &adj, std::unordered_map<int, int> &local_adj, int &V, int &E);
void gather_degrees(std::unordered_map<int, int> &local_degree, std::unordered_map<int, int> &global_degree, int procId, int npros);
std::vector<int> find_hub_nodes(const std::unordered_map<int, int> &global_degree);
std::pair<int, int> find_first_seed(const std::unordered_map<int, int> &global_degree);

#endif