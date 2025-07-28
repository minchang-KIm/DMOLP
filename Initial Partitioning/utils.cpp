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
    int total_size = 1;
    for (const auto &partition : partitions)
        total_size += 1 + partition.size();
    
    vector<int> buffer;
    buffer.reserve(total_size);
    buffer.push_back((int)partitions.size());

    for (const auto &partition : partitions) {
        buffer.push_back((int)partition.size());
        buffer.insert(buffer.end(), partition.begin(), partition.end());
    }

    return buffer;
}

vector<vector<int>> deserialize_partitions(const vector<int> &buffer) {
    if (buffer.empty()) return {};
    
    vector<vector<int>> partitions;
    int idx = 0;
    int num_partitions = buffer[idx++];
    partitions.reserve(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        int size = buffer[idx++];
        vector<int> partition(buffer.begin() + idx, buffer.begin() + idx + size);
        idx += size;
    }

    return partitions;
}

void sync_global_partitions(int procId, int nprocs, vector<vector<int>> &partitions, unordered_set<int> &global_partitioned) {
    vector<int> local_serialized = serialize_partitions(partitions);
    int local_size = local_serialized.size();

    vector<int> all_sizes(nprocs);
    MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displacements(nprocs);
    int total_size = 0;
    for (int p = 0; p < nprocs; p++) {
        displacements[p] = total_size;
        total_size += all_sizes[p];
    }

    vector<int> serialized(total_size);
    MPI_Allgatherv(local_serialized.data(), local_size, MPI_INT, serialized.data(), all_sizes.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    #pragma omp parallel
    {
        unordered_set<int> thread_partitioned;
        vector<vector<int>> thread_updates(partitions.size());

        #pragma omp for schedule(static)
        for (int p = 0; p < nprocs; p++) {
            if (p == procId || all_sizes[p] == 0) continue;

            vector<int> proc_data(serialized.begin() + displacements[p], serialized.begin() + displacements[p] + all_sizes[p]);
            vector<vector<int>> received_partitions = deserialize_partitions(proc_data);

            int recv_size = (int)received_partitions.size();
            int partitions_size = (int)partitions.size();

            for (int i = 0; i < recv_size && i < partitions_size; i++) {
                const vector<int> received_partition = received_partitions[i];
                for (int node : received_partition) {
                    thread_updates[i].push_back(node);
                    thread_partitioned.insert(node);
                }
            }
        }

        #pragma omp critical
        {
            const int updates_size = thread_updates.size();
            for (int i = 0;  i < updates_size; i++) {
                const vector<int> thread_update = thread_updates[i];
                for (int node : thread_update) {
                    if (global_partitioned.find(node) == global_partitioned.end()) {
                        partitions[i].push_back(node);
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