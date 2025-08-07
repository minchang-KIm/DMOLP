#include <mpi.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <iostream>
#include <mutex>
#include <chrono>

#include "graph_types.h"
#include "utils.hpp"
#include "phase1/seed.hpp"

using namespace std;

const int INF = INT_MAX;
const int MAX_LEVELS = 3;

BFSResult compute_bfs(int procId, int nprocs, int start_node, const unordered_map<int, vector<int>> &adj) {
    BFSResult result;

    unordered_set<int> visited;
    unordered_set<int> current_level;
    
    if (start_node % nprocs == procId) {
        current_level.insert(start_node);
        visited.insert(start_node);
    }

    int level = 0;

    while (level < MAX_LEVELS) {
        int local_has_nodes = current_level.empty() ? 0 : 1;
        int global_has_nodes = 0;
        MPI_Allreduce(&local_has_nodes, &global_has_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_has_nodes == 0) break;

        result.ensure_level(level);
        result.levels[level] = current_level;

        for (int node : current_level) {
            result.all_visited.insert(node);
        }

        vector<int> local_frontier(current_level.begin(), current_level.end());

        vector<int> recv_counts(nprocs);
        int send_count = local_frontier.size();
        MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        vector<int> displs(nprocs, 0);
        int total = 0;
        for (int i = 0; i < nprocs; ++i) {
            displs[i] = total;
            total += recv_counts[i];
        }

        vector<int> global_frontier(total);
        MPI_Allgatherv(local_frontier.data(), send_count, MPI_INT, global_frontier.data(), recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        unordered_set<int> next_level;
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

        unordered_set<int> temp_visited = visited;
        for (const auto &thread_neighbor : thread_neighbors) {
            for (int neighbor : thread_neighbor) {
                if (temp_visited.find(neighbor) == temp_visited.end()) {
                    visited.insert(neighbor);
                    temp_visited.insert(neighbor);
                    next_level.insert(neighbor);
                }
            }
        }

        current_level = move(next_level);
        ++level;
    }

    return result;
}

BFSResult gather_result(int procId, int nprocs, const BFSResult &local_result) {
    if (nprocs <= 1) return local_result;

    BFSResult global_result;

    vector<int> local_visited(local_result.all_visited.begin(), local_result.all_visited.end());
    int local_size = static_cast<int>(local_visited.size());
    
    vector<int> all_sizes(nprocs);
    MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displs(nprocs, 0);
    int total_size = 0;
    for (int i = 0; i < nprocs; i++) {
        displs[i] = total_size;
        total_size += all_sizes[i];
    }

    if (total_size > 0) {
        vector<int> all_visited(total_size);
        MPI_Allgatherv(local_visited.data(), local_size, MPI_INT, all_visited.data(), all_sizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
        global_result.all_visited.insert(all_visited.begin(), all_visited.end());
    }

    int local_max_level = static_cast<int>(local_result.levels.size());
    int global_max_level = 0;
    MPI_Allreduce(&local_max_level, &global_max_level, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    global_result.levels.resize(global_max_level);

    for (int level = 0; level < global_max_level; level++) {
        vector<int> local_level_nodes;
        if (level < static_cast<int>(local_result.levels.size())) local_level_nodes.assign(local_result.levels[level].begin(), local_result.levels[level].end());

        int local_level_size = static_cast<int>(local_level_nodes.size());
        vector<int> level_sizes(nprocs);
        MPI_Allgather(&local_level_size, 1, MPI_INT, level_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        vector<int> level_displs(nprocs, 0);
        int level_total = 0;
        for (int i = 0; i < nprocs; i++) {
            level_displs[i] = level_total;
            level_total += level_sizes[i];
        }

        if (level_total > 0) {
            vector<int> all_level_nodes(level_total);
            MPI_Allgatherv(local_level_nodes.data(), local_level_size, MPI_INT, all_level_nodes.data(), level_sizes.data(), level_displs.data(), MPI_INT, MPI_COMM_WORLD);
            global_result.levels[level].insert(all_level_nodes.begin(), all_level_nodes.end());
        }
    }

    return global_result;
}

int find_next_seed(int procId, int nprocs, const vector<int> &selected_seeds, const vector<int> &hub_nodes, const vector<bool> &used_hubs, const unordered_map<int, vector<int>> &adj, const unordered_map<int, int> &global_degree) {
    unordered_set<int> covered_nodes;
    static unordered_map<int, unordered_set<int>> bfs_cache;

    for (int seed : selected_seeds) {
        if (bfs_cache.find(seed) != bfs_cache.end()) {
            const auto &cached_nodes = bfs_cache[seed];
            covered_nodes.insert(cached_nodes.begin(), cached_nodes.end());
        } else {
            BFSResult local_bfs = compute_bfs(procId, nprocs, seed, adj);
            BFSResult global_bfs = gather_result(procId, nprocs, local_bfs);

            bfs_cache[seed] = global_bfs.all_visited;
            covered_nodes.insert(global_bfs.all_visited.begin(), global_bfs.all_visited.end());
        }
    }

    int local_best_hub = -1;
    int local_max_degree = -1;
    size_t hub_count = hub_nodes.size();

    for (size_t i = procId; i < hub_count; i += nprocs) {
        if (used_hubs[i]) continue;

        int hub = hub_nodes[i];
        if (covered_nodes.count(hub)) continue;

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

vector<int> find_seeds(int procId, int nprocs, int numParts, const pair<int, int> &first_seed, const vector<int> &hub_nodes, const unordered_map<int, int> &global_degree, const unordered_map<int, vector<int>> &adj) {
    auto total_start_time = chrono::high_resolution_clock::now();
    vector<int> selected_seeds = {first_seed.first};
    if (numParts <= 0 || hub_nodes.empty()) {
        if (procId == 0) cout << "Error: Invalid numParts (" << numParts << ") or empty hub_nodes" << endl;
        return selected_seeds;
    }
    
    if (procId == 0) cout << "Finding " << numParts << " seeds from " << hub_nodes.size() << " hub nodes using " << nprocs << " processes..." << endl;

    vector<bool> used_hubs(hub_nodes.size(), false);
    auto it = find(hub_nodes.begin(), hub_nodes.end(), first_seed.first);
    if (it != hub_nodes.end()) used_hubs[distance(hub_nodes.begin(), it)] = true;

    for (int k = 1; k < numParts; k++) {
        auto start_time = chrono::high_resolution_clock::now();

        int next_seed = find_next_seed(procId, nprocs, selected_seeds, hub_nodes, used_hubs, adj, global_degree);
        
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

        if (next_seed == -1) {
            if (procId == 0) cout << "Warning: Could not find valid seed " << (k + 1) << " outside existing coverage" << endl;
            break;
        }

        selected_seeds.push_back(next_seed);

        auto it = find(hub_nodes.begin(), hub_nodes.end(), next_seed);
        if (it != hub_nodes.end()) used_hubs[distance(hub_nodes.begin(), it)] = true;

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