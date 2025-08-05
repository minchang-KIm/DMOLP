#ifndef BFS_HPP
#define BFS_HPP

#include <vector>
#include <unordered_map>

std::unordered_map<int, int> batch_bfs(int source, int procId, int nprocs, const std::vector<int> &hub_nodes, const std::unordered_map<int, std::vector<int>> &local_adj);

#endif