#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include "utils.hpp"

using namespace std;

void sync_vector(int procId, int sourceProc, vector<int> &vec) {
    int size = (int)vec.size();
    MPI_Bcast(&size, 1, MPI_INT, sourceProc, MPI_COMM_WORLD);

    if (procId != sourceProc) vec.resize(size);
    MPI_Bcast(vec.data(), size, MPI_INT, sourceProc, MPI_COMM_WORLD);
}

void sync_unordered_map(int procId, int sourceProc, unordered_map<int, int> &map) {
    int size;
    vector<int> serialized_map;

    if (procId == sourceProc) {
        for (const auto &data : map) {
            serialized_map.push_back(data.first);
            serialized_map.push_back(data.second);
        }
    }

    size = (int)serialized_map.size();
    MPI_Bcast(&size, 1, MPI_INT, sourceProc, MPI_COMM_WORLD);

    if (procId != sourceProc) serialized_map.resize(size);
    MPI_Bcast(serialized_map.data(), size, MPI_INT, sourceProc, MPI_COMM_WORLD);

    if (procId != sourceProc) {
        for (int i = 0; i < size; i+=2) {
            int key = serialized_map[i];
            int value = serialized_map[i + 1];
            map[key] = value;
        }
    }
}

vector<int> serialize_node_info(const vector<NodeInfo> &nodes) {
    vector<int> buffer;

    buffer.push_back((int)nodes.size());
    for (const auto &s : nodes) {
        buffer.push_back(s.vertex);
        buffer.push_back((int)s.neighbors.size());
        buffer.insert(buffer.end(), s.neighbors.begin(), s.neighbors.end());
    }

    return buffer;
}

vector<NodeInfo> deserialize_node_info(const vector<int> &buffer) {
    vector<NodeInfo> nodes;
    int idx = 0;
    int num_nodes = buffer[idx++];

    for (int i = 0; i < num_nodes; ++i) {
        int node = buffer[idx++];
        int neighbor_counts = buffer[idx++];
        
        vector<int> neighbors(buffer.begin() + idx, buffer.begin() + idx + neighbor_counts);
        idx += neighbor_counts;

        nodes.push_back({node, neighbors});
    }

    return nodes;
}

vector<int> serialize_partitions(const vector<vector<int>> &partitions) {
    vector<int> buffer;
    buffer.push_back((int)partitions.size());

    for (const auto &partition : partitions) {
        buffer.push_back((int)partition.size());
        buffer.insert(buffer.end(), partition.begin(), partition.end());
    }

    return buffer;
}

vector<vector<int>> deserialize_partitions(const vector<int> &buffer) {
    vector<vector<int>> partitions;
    int idx = 0;
    int num_partitions = buffer[idx++];

    for (int i = 0; i < num_partitions; i++) {
        int size = buffer[idx++];
        vector<int> partition(buffer.begin() + idx, buffer.begin() + idx + size);
        idx += size;
        partitions.push_back(partition);
    }

    return partitions;
}

void sync_global_partitions(int procId, int nprocs, vector<vector<int>> &partitions, unordered_set<int> &global_partitioned) {
    for (int p = 0 ; p < nprocs; p++) {
        vector<int> serialized;
        int buffer_size = 0;

        if (procId == p) {
            serialized = serialize_partitions(partitions);
            buffer_size = (int)serialized.size();
        }

        MPI_Bcast(&buffer_size, 1, MPI_INT, p, MPI_COMM_WORLD);

        if (procId != p) serialized.resize(buffer_size);

        if (buffer_size > 0) {
            MPI_Bcast(serialized.data(), buffer_size, MPI_INT, p, MPI_COMM_WORLD);

            if (procId != p) {
                vector<vector<int>> received_partitions = deserialize_partitions(serialized);

                for (int i = 0;  i < (int)received_partitions.size() && i < (int)partitions.size(); i++) {
                    for (int node : received_partitions[i]) {
                        if (find(partitions[i].begin(), partitions[i].end(), node) == partitions[i].end()) partitions[i].push_back(node);
                        global_partitioned.insert(node);
                    }
                }
            }
        }
    }
}

void seed_redistribution(int procId, int nprocs, int numParts, const vector<int> &remaining_seeds, const unordered_map<int, vector<int>> &local_adj, vector<vector<int>> &partitions, unordered_set<int> &global_partitioned) {
    if (remaining_seeds.empty()) return;

    vector<NodeInfo> remaining_seed_infos;
    for (int seed : remaining_seeds) {
        auto it = local_adj.find(seed);
        vector<int> neighbors = (it != local_adj.end()) ? it->second : vector<int>();
        remaining_seed_infos.push_back({seed, neighbors});
    }
    
    vector<NodeInfo> all_remaining_seeds;

    if (procId == 0) {
        all_remaining_seeds = remaining_seed_infos;

        for (int p = 1; p < nprocs; p++) {
            int recv_size = 0;
            MPI_Recv(&recv_size, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (recv_size > 0) {
                vector<int> recvbuf(recv_size);
                MPI_Recv(recvbuf.data(), recv_size, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                vector<NodeInfo> proc_seeds = deserialize_node_info(recvbuf);
                all_remaining_seeds.insert(all_remaining_seeds.end(), proc_seeds.begin(), proc_seeds.end());
            }
        }

        int seed_idx = 0;
        for (int p = 0; p < numParts && seed_idx < (int)all_remaining_seeds.size(); p++) {
            if (partitions[p].empty()) {
                partitions[p].push_back(all_remaining_seeds[seed_idx].vertex);
                global_partitioned.insert(all_remaining_seeds[seed_idx].vertex);
                seed_idx++;
            }
        }
    } else {
        vector<int> sendbuf = serialize_node_info(remaining_seed_infos);
        int send_size = (int)sendbuf.size();

        MPI_Send(&send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (send_size > 0) MPI_Send(sendbuf.data(), send_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
}