#include <mpi.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <climits>
#include <iostream>
#include <chrono>
#include <boost/dynamic_bitset.hpp>

#include "graph_types.h"
#include "utils.hpp"
#include "phase1/seed.hpp"

using namespace std;

const int INF = INT_MAX;
const int MAX_LEVELS = 3;

BFSResult compute_bfs(int procId, int nprocs, int start_node, const unordered_map<int, vector<int>> &adj, size_t num_nodes) {
    BFSResult result(num_nodes);

    boost::dynamic_bitset<> visited(num_nodes);
    boost::dynamic_bitset<> current_level(num_nodes);
    
    if (start_node % nprocs == procId) {
        current_level.set(start_node);
        visited.set(start_node);
    }

    int level = 0;

    while (level < MAX_LEVELS) {
        int local_has_nodes = current_level.any() ? 1 : 0;
        int global_has_nodes = 0;
        MPI_Allreduce(&local_has_nodes, &global_has_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_has_nodes == 0) break;

        result.ensure_level(level, num_nodes);
        result.levels[level] |= current_level;
        result.all_visited |= current_level;

        vector<int> local_frontier;
        for (size_t i = current_level.find_first(); i != boost::dynamic_bitset<>::npos; i = current_level.find_next(i)) {
            local_frontier.push_back(static_cast<int>(i));
        }

        vector<int> recv_counts(nprocs);
        int send_count = static_cast<int>(local_frontier.size());
        MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        vector<int> displs(nprocs, 0);
        int total = 0;
        for (int i = 0; i < nprocs; ++i) {
            displs[i] = total;
            total += recv_counts[i];
        }

        vector<int> global_frontier(total);
        MPI_Allgatherv(local_frontier.data(), send_count, MPI_INT, global_frontier.data(), recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        boost::dynamic_bitset<> next_level(num_nodes);
        vector<vector<int>> thread_neighbors(omp_get_max_threads());

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            vector<int> &local_neighbors = thread_neighbors[tid];

            #pragma omp for schedule(dynamic, 64) nowait
            for (int i = 0; i < static_cast<int>(global_frontier.size()); ++i) {
                int node = global_frontier[i];

                if (node % nprocs != procId) continue;

                auto it = adj.find(node);
                if (it != adj.end()) {
                    for (int neighbor : it->second) {
                        local_neighbors.push_back(neighbor);
                    }
                }
            }
        }

        boost::dynamic_bitset<> temp_visited = visited;
        for (const auto &thread_neighbor : thread_neighbors) {
            for (int neighbor : thread_neighbor) {
                if (!temp_visited[neighbor]) {
                    visited.set(neighbor);
                    temp_visited.set(neighbor);
                    next_level.set(neighbor);
                }
            }
        }

        current_level = boost::move(next_level);
        ++level;
    }

    return result;
}

BFSResult gather_result(int procId, int nprocs, const BFSResult &local_result, size_t num_nodes) {
    if (nprocs <= 1) return local_result;

    BFSResult global_result(num_nodes);

    vector<unsigned long> local_bits(local_result.all_visited.num_blocks());
    boost::to_block_range(local_result.all_visited, local_bits.begin());

    size_t local_bits_size = local_bits.size();
    vector<unsigned long> global_bits(local_bits_size);
    MPI_Allreduce(local_bits.data(), global_bits.data(), local_bits_size, MPI_UNSIGNED_LONG, MPI_BOR, MPI_COMM_WORLD);
    boost::from_block_range(global_bits.begin(), global_bits.end(), global_result.all_visited);

    int local_max_level = static_cast<int>(local_result.levels.size());
    int global_max_level = 0;
    MPI_Allreduce(&local_max_level, &global_max_level, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    global_result.levels.resize(global_max_level, boost::dynamic_bitset<>(num_nodes));

    for (int level = 0; level < global_max_level; level++) {
        boost::dynamic_bitset<> local_level(num_nodes);
        if (level < static_cast<int>(local_result.levels.size())) local_level = local_result.levels[level];

        vector<unsigned long> local_level_bits(local_level.num_blocks());
        boost::to_block_range(local_level, local_level_bits.begin());

        size_t local_level_bits_size = local_level_bits.size();
        vector<unsigned long> global_level_bits(local_level_bits_size);
        MPI_Allreduce(local_level_bits.data(), global_level_bits.data(), local_level_bits_size, MPI_UNSIGNED_LONG, MPI_BOR, MPI_COMM_WORLD);
        boost::from_block_range(global_level_bits.begin(), global_level_bits.end(), global_result.levels[level]);
    }

    return global_result;
}

int find_next_seed(int procId, int nprocs, const vector<int> &selected_seeds, const vector<int> &hub_nodes, const boost::dynamic_bitset<> &used_hubs, const unordered_map<int, vector<int>> &adj, const unordered_map<int, int> &global_degree, size_t num_nodes) {
    boost::dynamic_bitset<> covered_nodes(num_nodes);
    static unordered_map<int, boost::dynamic_bitset<>> bfs_cache;

    for (int seed : selected_seeds) {
        if (bfs_cache.find(seed) != bfs_cache.end()) {
            covered_nodes |= bfs_cache[seed];
        } else {
            BFSResult local_bfs = compute_bfs(procId, nprocs, seed, adj, num_nodes);
            BFSResult global_bfs = gather_result(procId, nprocs, local_bfs, num_nodes);

            bfs_cache[seed] = global_bfs.all_visited;
            covered_nodes |= global_bfs.all_visited;
        }
    }

    int local_best_hub = -1;
    int local_max_degree = -1;
    size_t hub_count = hub_nodes.size();

    for (size_t i = procId; i < hub_count; i += nprocs) {
        if (used_hubs[i]) continue;

        int hub = hub_nodes[i];
        if (covered_nodes[hub]) continue;

        int degree = global_degree.count(hub) ? global_degree.at(hub) : 0;

        if (degree > local_max_degree) {
            local_max_degree = degree;
            local_best_hub = hub;
        }
    }

    vector<int> all_degrees(nprocs) , all_hubs(nprocs);

    MPI_Allgather(&local_max_degree, 1, MPI_INT, all_degrees.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local_best_hub, 1, MPI_INT, all_hubs.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int global_max_degree = -1;
    int global_best_hub = -1;

    for (int i = 0; i < nprocs; i++) {
        if (all_degrees[i] > global_max_degree) {
            global_max_degree = all_degrees[i];
            global_best_hub = all_hubs[i];
        }
    }

    if (procId == 0 && global_best_hub != -1) cout << "Found next seed " << global_best_hub << " with degree " << global_max_degree << " (outside existing coverage)" << endl;

    return global_best_hub;
}

vector<int> find_seeds(int procId, int nprocs, int numParts, size_t num_nodes, const pair<int, int> &first_seed, const vector<int> &hub_nodes, const unordered_map<int, int> &global_degree, const unordered_map<int, vector<int>> &adj) {
    auto total_start_time = chrono::high_resolution_clock::now();
    vector<int> selected_seeds = {first_seed.first};
    if (numParts <= 0 || hub_nodes.empty()) {
        if (procId == 0) cout << "Error: Invalid numParts (" << numParts << ") or empty hub_nodes" << endl;
        return selected_seeds;
    }
    
    if (procId == 0) cout << "Finding " << numParts << " seeds from " << hub_nodes.size() << " hub nodes using " << nprocs << " processes..." << endl;

    boost::dynamic_bitset<> used_hubs(hub_nodes.size());
    auto it = find(hub_nodes.begin(), hub_nodes.end(), first_seed.first);
    if (it != hub_nodes.end()) used_hubs.set(distance(hub_nodes.begin(), it));

    for (int k = 1; k < numParts; k++) {
        auto start_time = chrono::high_resolution_clock::now();

        int next_seed = find_next_seed(procId, nprocs, selected_seeds, hub_nodes, used_hubs, adj, global_degree, num_nodes);
        
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

        if (next_seed == -1) {
            if (procId == 0) cout << "Warning: Could not find valid seed " << (k + 1) << " outside existing coverage" << endl;
            break;
        }

        selected_seeds.push_back(next_seed);

        auto it = find(hub_nodes.begin(), hub_nodes.end(), next_seed);
        if (it != hub_nodes.end()) used_hubs.set(distance(hub_nodes.begin(), it));

        int next_degree = global_degree.count(next_seed) ? global_degree.at(next_seed) : 0;
        if (procId == 0) cout << "Selected seed " << (k + 1) << ": " << next_seed << " with degree " << next_degree << " (took " << duration.count() << " ms)\n" << endl;
    }
    auto total_end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(total_end_time - total_start_time);

    size_t result_size = selected_seeds.size();
    if (procId == 0) {
        cout << "Seed selection completed.\n";
        cout << "Selected " << result_size << " seed: ";
        for (size_t i = 0; i < result_size; i++) {
            cout << selected_seeds[i];
            if (i < result_size - 1) cout << ", ";
        }
        cout << "\nTotal execution time: " << total_duration.count() << "(ms)" << endl;
    }
    
    return selected_seeds;
}