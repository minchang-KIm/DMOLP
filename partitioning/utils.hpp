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
}

void sync_vector(int procId, int sourceProc, std::vector<int> &vec);
std::vector<int> serialize_node_info(const std::vector<NodeInfo> &nodes);
std::vector<NodeInfo> deserialize_node_info(const std::vector<int> &buffer);
std::vector<int> serialize_partitions(const std::vector<std::vector<int>> &partitions);
std::vector<std::vector<int>> deserialize_partitions(const std::vector<int> &buffer);
void sync_global_partitions(int procId, int nprocs, std::vector<std::vector<int>> &partitions, std::unordered_set<int> &global_partitioned);
void seed_redistribution(int procId, int nprocs, int numParts, const std::vector<int> &remaining_seeds, const std::unordered_map<int, std::vector<int>> &local_adj, std::vector<std::vector<int>> &partitions, std::unordered_set<int> &global_partitioned);