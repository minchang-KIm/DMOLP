#include <vector>
#include <unordered_map>

void load_graph(const char *filename, int procId, int nprocs, std::unordered_map<int, std::vector<int>> &adj, std::unordered_map<int, int> &local_adj);
void gather_degrees(std::unordered_map<int, int> &local_degree, std::unordered_map<int, int> &global_degree, int procId, int npros);