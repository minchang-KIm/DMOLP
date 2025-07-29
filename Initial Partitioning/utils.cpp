#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>

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
    if (buffer.empty()) return nodes;

    int idx = 0;
    int num_nodes = buffer[idx++];
    nodes.reserve(num_nodes);

    for (int i = 0; i < num_nodes; ++i) {
        int node = buffer[idx++];
        int neighbor_counts = buffer[idx++];
        
        vector<int> neighbors(buffer.begin() + idx, buffer.begin() + idx + neighbor_counts);
        idx += neighbor_counts;

        nodes.push_back({node, neighbors});
    }

    return nodes;
}

vector<int> serialize_updates(const vector<PartitionUpdate> &updates) {
    vector<int> buffer;
    buffer.push_back((int)updates.size());

    for (const auto &update : updates) {
        buffer.push_back(update.partition_id);
        buffer.push_back(update.node);
    }

    return buffer;
}

vector<PartitionUpdate> deserialize_updates(const vector<int> &buffer) {
    vector<PartitionUpdate> updates;
    if (buffer.empty()) return updates;

    int idx = 0;
    int num_updates = buffer[idx++];
    updates.reserve(num_updates);

    for (int i = 0; i < num_updates; i++) {
        int partition_id = buffer[idx++];
        int node = buffer[idx++];
        updates.emplace_back(partition_id, node);
    }

    return updates;
}

void sync_updates(int procId, int nprocs, const vector<PartitionUpdate> &local_updates, vector<vector<int>> &partitions, unordered_set<int> &global_partitioned) {
    vector<int> local_serialized = serialize_updates(local_updates);
    int local_size = (int)local_serialized.size();

    vector<int> all_sizes(nprocs);
    MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displacements(nprocs);
    int total_size = 0;
    for (int p = 0; p < nprocs; p++) {
        displacements[p] = total_size;
        total_size += all_sizes[p];
    }

    vector<int> total_serialized(total_size);
    MPI_Allgatherv(local_serialized.data(), local_size, MPI_INT, total_serialized.data(), all_sizes.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    for (int p = 0; p < nprocs; p++) {
        if (all_sizes[p] == 0) continue;

        vector<int> proc_data(total_serialized.begin() + displacements[p], total_serialized.begin() + displacements[p] + all_sizes[p]);
        vector<PartitionUpdate> proc_updates = deserialize_updates(proc_data);

        for (const auto &update : proc_updates) {
            if (global_partitioned.find(update.node) == global_partitioned.end()) {
                partitions[update.partition_id].push_back(update.node);
                global_partitioned.insert(update.node);
            }
        }
    }
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
    if (buffer.empty()) return partitions;

    int idx = 0;
    int num_partitions = buffer[idx++];
    partitions.resize(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        int size = buffer[idx++];
        vector<int> partition(buffer.begin() + idx, buffer.begin() + idx + size);
        idx += size;
    }

    return partitions;
}

void sync_global_partitions(int procId, int nprocs, vector<vector<int>> &partitions, unordered_set<int> &global_partitioned, vector<PartitionUpdate> &pending_updates) {
    if (pending_updates.empty()) return;

    sync_updates(procId, nprocs, pending_updates, partitions, global_partitioned);

    pending_updates.clear();
}

void add_node_to_partition(int node, int partition_id, vector<vector<int>> &partitions, unordered_set<int> &global_partitioned, vector<PartitionUpdate> &pending_updates) {
    if (global_partitioned.find(node) == global_partitioned.end()) {
        partitions[partition_id].push_back(node);
        global_partitioned.insert(node);
        pending_updates.emplace_back(partition_id, node);
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

        size_t seed_idx = 0;
        size_t remaining_size = all_remaining_seeds.size();
        for (int p = 0; p < numParts && seed_idx < remaining_size; p++) {
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

void print_summary(int procId, int nprocs, const vector<vector<int>> &partitions, const unordered_map<int, int> &global_degree) {
    if (procId != 0) return;

    cout << "Number of partitions: " << partitions.size() << "\n";
    cout << "Total nodes in graph: " << global_degree.size() << "\n";

    int total_partitioned = 0;
    size_t min_size = INT64_MAX;
    size_t max_size = 0;
    size_t partitions_size = partitions.size();

    for (size_t i = 0; i < partitions_size; i++) {
        size_t size = partitions[i].size();
        total_partitioned += size;
        min_size = min(min_size, size);
        max_size = max(max_size, size);

        cout << "Partition " << i << ": " << size << " nodes\n";
    }

    cout << "Total partitioned nodes: " << total_partitioned << "\n";
    cout << "Min partition size: " << min_size << "\n";
    cout << "Max partition size: " << max_size << "\n";
    
    if (total_partitioned > 0) {
        double avg_size = (double)total_partitioned / partitions.size();
        double imbalance = (double)max_size / avg_size;
        cout << "Average partition size: " << avg_size << "\n";
        cout << "Load imbalance ratio: " << imbalance << "\n";
    }
    
    cout << "=======================================\n\n";
}