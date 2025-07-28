#include <mpi.h>
#include <omp.h>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <iostream>

#include "utils.hpp"
#include "partition.hpp"

using namespace std;

double compute_ratio(int target, const vector<int> &neighbors, const unordered_set<int> &partition_set, const unordered_map<int, int> &global_degree) {
    int node_degree = global_degree.at(target);
    if (node_degree == 1) return 1.0;

    int partition_degree = 0;
    for (int neighbor : neighbors)
        if (partition_set.count(neighbor)) partition_degree++;

    return (double)partition_degree / node_degree;
}

vector<FrontierNode> collect_frontiers(int procId, int nprocs, const vector<int> &current_partition, const unordered_map<int, vector<int>> &local_adj, const unordered_map<int, int> &global_degree, const unordered_set<int> &global_partitioned, int partition_id = -1) {
    vector<FrontierNode> local_frontiers;
    unordered_set<int> frontier_candidates;
    unordered_set<int> partition_set(current_partition.begin(), current_partition.end());
    vector<int> boundary;

    #pragma omp parallel
    {
        vector<int> thread_boundary;

        #pragma omp for
        for (int i = 0; i < (int)current_partition.size(); i++) {
            int v = current_partition[i];
            auto it = local_adj.find(v);
            if (it == local_adj.end()) continue;

            bool is_boundary = false;
            for (int neighbor : it->second) {
                if (partition_set.find(neighbor) == partition_set.end()) {
                    is_boundary = true;
                    break;
                }
            }

            if (is_boundary) thread_boundary.push_back(v);
        }

        #pragma omp critical
        {
            boundary.insert(boundary.end(), thread_boundary.begin(), thread_boundary.end());
        }
    }

    if (procId == 0) cout << "[proc " << procId << "]     Boundary nodes: " << boundary.size() << "/" << current_partition.size() << "\n";

    #pragma omp parallel
    {
        unordered_set<int> thread_candidates;

        #pragma omp for
        for (int i = 0; i < (int)boundary.size(); i++) {
            int v = boundary[i];
            auto it = local_adj.find(v);
            if (it == local_adj.end()) continue;

            for (int neighbor : it->second) {
                if (global_partitioned.find(neighbor) == global_partitioned.end()) {
                    thread_candidates.insert(neighbor);
                }
            }
        }

        #pragma omp critical
        {
            frontier_candidates.insert(thread_candidates.begin(), thread_candidates.end());
        }
    }

    vector<int> candidates_vec(frontier_candidates.begin(), frontier_candidates.end());

    #pragma omp parallel
    {
        vector<FrontierNode> thread_frontiers;

        #pragma omp for
        for (int i = 0; i < (int)candidates_vec.size(); i++) {
            int u = candidates_vec[i];
            auto adj_it = local_adj.find(u);
            if (adj_it != local_adj.end()) {
                double ratio = compute_ratio(u, adj_it->second, partition_set, global_degree);

                int partition_degree = 0;
                for (int neighbor : adj_it->second) {
                    if (find(current_partition.begin(), current_partition.end(), neighbor) != current_partition.end()) {
                        partition_degree++;
                    }
                }
                
                thread_frontiers.push_back(FrontierNode(u, ratio, partition_degree, global_degree.at(u), partition_id));
            }
        }

        #pragma omp critical
        {
            local_frontiers.insert(local_frontiers.end(), thread_frontiers.begin(), thread_frontiers.end());
        }
    }

    return local_frontiers;
}

unordered_map<int, int> resolve_concurrency(const vector<FrontierNode> &candidates) {
    unordered_map<int, vector<FrontierNode>> node_to_candidates;
    unordered_map<int, int> node_to_partition;

    for (const auto &candidate : candidates)
        node_to_candidates[candidate.vertex].push_back(candidate);

    for (const auto &[node, node_candidates] : node_to_candidates) {
        if (node_candidates.size() == 1) node_to_partition[node] = node_candidates[0].partition_id;
        else {
            const FrontierNode *best_candidate = nullptr;
            for (const auto &candidate : node_candidates)
                if (best_candidate == nullptr || candidate > *best_candidate || (abs(candidate.ratio - best_candidate->ratio) < 1e-9 && candidate.partition_degree > best_candidate->partition_degree)) best_candidate = &candidate;
            node_to_partition[node] = best_candidate->partition_id;
        }
    }

    return node_to_partition;
}

vector<FrontierNode> gather_frontiers(int procId, int nprocs, const vector<FrontierNode> &local_frontiers) {
    vector<int> sendbuf;
    sendbuf.push_back((int)local_frontiers.size());
    for (const auto &frontier : local_frontiers) {
        sendbuf.push_back(frontier.vertex);
        sendbuf.push_back(frontier.partition_degree);
        sendbuf.push_back(frontier.total_degree);
        sendbuf.push_back((int)(frontier.ratio * 1000000));
    }

    vector<int> recv_counts(nprocs);
    int send_size = (int)sendbuf.size();
    MPI_Allgather(&send_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    vector<int> displacements(nprocs);
    int total_size = 0;
    for (int p = 0; p < nprocs; p++) {
        displacements[p] = total_size;
        total_size += recv_counts[p];
    }

    vector<int> recvbuf(total_size);
    MPI_Allgatherv(sendbuf.data(), send_size, MPI_INT, recvbuf.data(), recv_counts.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    vector<FrontierNode> all_frontiers;
    int idx = 0;
    for (int p = 0; p < nprocs; p++) {
        int num_frontiers = recvbuf[idx++];
        for (int i = 0; i < num_frontiers; i++) {
            FrontierNode frontier;
            frontier.vertex = recvbuf[idx++];
            frontier.partition_degree = recvbuf[idx++];
            frontier.total_degree = recvbuf[idx++];
            frontier.ratio = (double)recvbuf[idx++] / 1000000.0;
            all_frontiers.push_back(frontier);
        }
    }
    
    return all_frontiers;
}

void partition_expansion(int procId, int nprocs, int numParts, int theta, const vector<int> &seeds, const unordered_map<int, int> &global_degree, const unordered_map<int, vector<int>> &local_adj, vector<vector<int>> &partitions) {
    unordered_set<int> all_nodes;
    for (const auto &[node, degree] : global_degree)
        all_nodes.insert(node);

    vector<unordered_set<int>> global_partitions(numParts);
    unordered_set<int> global_partitioned;
    partitions.resize(numParts);

    if (procId == 0) cout << "[proc " << procId << "] Seeding partitions...\n";

    vector<NodeInfo> available_seeds;
    vector<int> used_seeds;

    for (int seed : seeds) {
        auto it = local_adj.find(seed);
        if (it != local_adj.end()) {
            available_seeds.push_back({seed, it->second});
            used_seeds.push_back(seed);
        }
    }

    for (int i = 0; i < min((int)available_seeds.size(), numParts); i++) {
        partitions[i].push_back(available_seeds[i].vertex);
        global_partitioned.insert(available_seeds[i].vertex);
    }

    if (procId == 0) cout << "[proc " << procId << "] Local seeds assigned: " << available_seeds.size() << "/" << seeds.size() << "\n";

    vector<int> remaining_seeds;
    for (int seed : seeds)
        if (find(used_seeds.begin(), used_seeds.end(), seed) == used_seeds.end()) remaining_seeds.push_back(seed);

    seed_redistribution(procId, nprocs, numParts, remaining_seeds, local_adj, partitions, global_partitioned);
    sync_global_partitions(procId, nprocs, partitions, global_partitioned);

    if (procId == 0) cout << "[proc " << procId << "] Starting iterative expansion...\n";

    int iteration = 0;
    while (global_partitioned.size() < all_nodes.size()) {
        iteration++;
        if (procId == 0) cout << "[proc " << procId << "] Iteration " << iteration << " - Partitioned: " << global_partitioned.size() << "/" << all_nodes.size() << "\n";
        
        bool expansion = false;
        vector<FrontierNode> all_candidates;

        #pragma omp parallel
        {
            vector<FrontierNode> thread_candidates;

            #pragma omp for
            for (int p = 0; p < numParts; p++) {
                if (partitions[p].empty()) continue;

                vector<FrontierNode> frontiers = collect_frontiers(procId, nprocs, partitions[p], local_adj, global_degree, global_partitioned, p);
                sort(frontiers.begin(), frontiers.end(), greater<FrontierNode>());

                int max_add = min({(int)frontiers.size(), theta, (int)(all_nodes.size() - global_partitioned.size())});
                for (int i = 0; i < max_add; i++) {
                    const auto &frontier = frontiers[i];
                    if (global_partitioned.find(frontier.vertex) == global_partitioned.end()) thread_candidates.push_back(frontier);
                }
            }

            #pragma omp critical
            {
                all_candidates.insert(all_candidates.end(), thread_candidates.begin(), thread_candidates.end());
            }
        }

        if (!all_candidates.empty()) {
            unordered_map<int, int> node_assignments = resolve_concurrency(all_candidates);

            unordered_map<int, int> partition_add_count;
            for (int p = 0; p < numParts; p++)
                partition_add_count[p] = 0;

            for (const auto &[node, partition_id] : node_assignments) {
                partitions[partition_id].push_back(node);
                global_partitioned.insert(node);
                partition_add_count[partition_id]++;
                expansion = true;
            }

            if (procId == 0) {
                for (int p = 0; p < numParts; p++)
                    if (partition_add_count[p] > 0) cout << "[proc " << procId << "] Partition " << p << " added " << partition_add_count[p] << " nodes (new size: " << partitions[p].size() << ")\n";
            }
        }

        if (!expansion && global_partitioned.size() < all_nodes.size()) {
            vector<pair<int, int>> partition_sizes;
            for (int p = 0; p < numParts; p++)
                partition_sizes.push_back({(int)partitions[p].size(), p});
            sort(partition_sizes.begin(), partition_sizes.end());

            int min_partition = partition_sizes[0].second;

            for (int node : all_nodes) {
                if (global_partitioned.find(node) == global_partitioned.end()) {
                    partitions[min_partition].push_back(node);
                    global_partitioned.insert(node);

                    if (procId == 0) cout << "[proc " << procId << "] Forced assignment: node " << node << " to partition " << min_partition << "\n";
                    break;
                }
            }

            expansion = true;
        }

        sync_global_partitions(procId, nprocs, partitions, global_partitioned);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (procId == 0) {
        cout << "[proc " << procId << "] Partition expansion completed after " << iteration << " iterations\n";
        for (int p = 0; p < numParts; p++)
            cout << "[proc " << procId << "] Partition " << p << " final size: " << partitions[p].size() << "\n";
    }
}