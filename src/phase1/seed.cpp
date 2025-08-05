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

BFSResult compute_bfs(int procId, int nprocs, int start_node, const unordered_map<int, vector<int>> &adj, const unordered_set<int> *target_nodes = nullptr) {
    BFSResult result;

    unordered_set<int> visited;
    unordered_set<int> current_level;
    unordered_set<int> remaining_targets;
    bool has_targets = (target_nodes != nullptr && !target_nodes->empty());
    
    int local_has_targets = has_targets ? 1 : 0;
    int global_has_targets = 0;
    MPI_Allreduce(&local_has_targets, &global_has_targets, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    has_targets = (global_has_targets > 0);
    
    if (has_targets && target_nodes != nullptr) {
        remaining_targets = *target_nodes;
    }

    if (start_node % nprocs == procId) {
        current_level.insert(start_node);
        visited.insert(start_node);
    }

    const int MAX_LEVELS = 100000;
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
            if (has_targets) remaining_targets.erase(node);
        }

        if (has_targets) {
            int local_targets_remaining = remaining_targets.size();
            int global_targets_remaining = 0;
            MPI_Allreduce(&local_targets_remaining, &global_targets_remaining, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (global_targets_remaining == 0) break;
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

        for (const auto &thread_neighbor : thread_neighbors) {
            for (int neighbor : thread_neighbor) {
                if (!visited.count(neighbor)) {
                    visited.insert(neighbor);
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

    vector<int> all_visited(total_size);
    MPI_Allgatherv(local_visited.data(), local_size, MPI_INT, all_visited.data(), all_sizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    for (int node : all_visited) {
        global_result.all_visited.insert(node);
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

        vector<int> all_level_nodes(level_total);
        MPI_Allgatherv(local_level_nodes.data(), local_level_size, MPI_INT, all_level_nodes.data(), level_sizes.data(), level_displs.data(), MPI_INT, MPI_COMM_WORLD);

        for (int node : all_level_nodes) {
            global_result.levels[level].insert(node);
        }
    }

    return global_result;
}

unordered_map<int, int> extract_distances(const BFSResult &bfs_result, const vector<int> &target_nodes) {
    unordered_map<int, int> distances;

    for (int node : target_nodes) {
        distances[node] = INF;
    }

    int result_size = static_cast<int>(bfs_result.levels.size());
    for (int level = 0; level < result_size; level++) {
        for (int node : bfs_result.levels[level]) {
            if (distances.count(node) && distances[node] == INF) distances[node] = level;
        }
    }

    return distances;
}

vector<unordered_map<int, int>> compute_landmark_distances(int procId, int nprocs, const vector<int> &landmarks, const vector<int> &hub_nodes, const unordered_map<int, vector<int>> &adj) {
    size_t landmarks_size = landmarks.size();
    vector<unordered_map<int, int>> all_distances(landmarks_size);
    unordered_set<int> hub_set(hub_nodes.begin(), hub_nodes.end());

    if (procId == 0) cout << "Computing distances from " << landmarks.size() << " landmarks to " << hub_nodes.size() << " hub nodes using " << omp_get_max_threads() << " threads..." << endl;
    
    for (size_t i = 0; i < landmarks_size; i++) {
        int landmark = landmarks[i];

        if (procId == 0) cout << "Processing landmark " << landmark << " (" << (i+1) << "/" << landmarks.size() << ")" << endl;

        BFSResult local_bfs = compute_bfs(procId, nprocs, landmark, adj, &hub_set);
        BFSResult global_bfs = gather_result(procId, nprocs, local_bfs);

        all_distances[i] = extract_distances(global_bfs, hub_nodes);
        
        if (procId == 0) cout << "Completed landmark " << landmark << endl;
    }

    return all_distances;
}

pair<int, int> find_max_distance_hub(int procId, int nprocs, const vector<unordered_map<int, int>> &distances, const vector<int> &hub_nodes) {
    int local_max_distance = -1;
    int local_best_hub = -1;

    size_t hub_count = hub_nodes.size();
    size_t start_idx = (hub_count * procId) / nprocs;
    size_t end_idx = (hub_count * (procId + 1)) / nprocs;

    for (const auto &dist_map : distances) {
        for (size_t j = start_idx; j < end_idx; j++) {
            int hub = hub_nodes[j];
            auto it = dist_map.find(hub);
            if (it != dist_map.end() && it->second != INF && it->second > local_max_distance) {
                local_max_distance = it->second;
                local_best_hub = hub;
            }
        }
    }

    struct {
        int distance;
        int hub;
    } local_result = {local_max_distance, local_best_hub};
    
    struct {
        int distance;
        int hub;
    } global_result;

    MPI_Allreduce(&local_result, &global_result, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

    return make_pair(global_result.hub, global_result.distance);
}

unordered_map<int, int> compute_distances(int procId, int nprocs, int source_node, const vector<int> &hub_nodes, const unordered_map<int, vector<int>> &adj) {
    unordered_set<int> hub_set(hub_nodes.begin(), hub_nodes.end());

    BFSResult local_bfs = compute_bfs(procId, nprocs, source_node, adj, &hub_set);
    BFSResult global_bfs = gather_result(procId, nprocs, local_bfs);

    return extract_distances(global_bfs, hub_nodes);
}

int find_farthest_hub(int procId, int nprocs, const vector<int> &selected_seeds, const vector<int> &hub_nodes, const vector<bool> &used_hubs, const unordered_map<int, vector<int>> &adj, unordered_map<int, unordered_map<int, int>> &seed_to_hub) {
    if (procId == 0) cout << "Finding farthest hub from " << selected_seeds.size() << " selected seeds using " << nprocs << " processes..." << endl;
    
    for (int seed : selected_seeds) {
        if (seed_to_hub.find(seed) == seed_to_hub.end()) {
            if (procId == 0) cout << "Computing distances from new seed " << seed << " to all hub nodes..." << endl;
            
            unordered_set<int> hub_set(hub_nodes.begin(), hub_nodes.end());
            BFSResult local_bfs = compute_bfs(procId, nprocs, seed, adj, &hub_set);
            BFSResult global_bfs = gather_result(procId, nprocs, local_bfs);

            seed_to_hub[seed] = extract_distances(global_bfs, hub_nodes);
            
            if (procId == 0) cout << "Completed distances from seed " << seed << endl;
        }
    }

    int local_best_hub = -1;
    int local_max_distance = -1;
    
    size_t hub_count = hub_nodes.size();
    size_t start_idx = (hub_count * procId) / nprocs;
    size_t end_idx = (hub_count * (procId + 1)) / nprocs;

    for (size_t i = start_idx; i < end_idx; i++) {
        if (used_hubs[i]) continue;

        int hub = hub_nodes[i];
        int min_distance = INF;

        for (int seed : selected_seeds) {
            auto seed_distances_it = seed_to_hub.find(seed);
            if (seed_distances_it != seed_to_hub.end()) {
                auto hub_distance_it = seed_distances_it->second.find(hub);
                if (hub_distance_it != seed_distances_it->second.end() && hub_distance_it->second != INF) min_distance = min(min_distance, hub_distance_it->second);
            }
        }

        if (min_distance != INF && min_distance > local_max_distance) {
            local_max_distance = min_distance;
            local_best_hub = hub;
        }
    }

    struct {
        int distance;
        int hub;
    } local_result = {local_max_distance, local_best_hub};
    
    struct {
        int distance;
        int hub;
    } global_result;

    MPI_Allreduce(&local_result, &global_result, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (procId == 0 && global_result.hub != -1) {
        cout << "Found best hub " << global_result.hub << " with min distance " << global_result.distance << endl;
    }
    
    return global_result.hub;
}

vector<int> find_seeds(int procId, int nprocs, int numParts, const vector<int> &landmarks, const vector<int> &hub_nodes, const unordered_map<int, vector<int>> &adj) {
    vector<int> result;

    if (numParts <= 0 || hub_nodes.empty()) {
        if (procId == 0) cout << "Error: Invalid numParts (" << numParts << ") or empty hub_nodes" << endl;
        return result;
    }
    
    if (procId == 0) cout << "Finding " << numParts << " seeds from " << hub_nodes.size() << " hub nodes using " << nprocs << " processes..." << endl;
    
    vector<unordered_map<int, int>> all_distances = compute_landmark_distances(procId, nprocs, landmarks, hub_nodes, adj);
    
    auto [first_seed, max_dist] = find_max_distance_hub(procId, nprocs, all_distances, hub_nodes);

    if (first_seed == -1) {
        if (procId == 0) cout << "Error: No valid hub node found as first seed" << endl;
        return result;
    }

    vector<int> selected_seeds = {first_seed};
    vector<bool> used_hubs(hub_nodes.size(), false);
    unordered_map<int, unordered_map<int, int>> seed_to_hub;

    auto it = find(hub_nodes.begin(), hub_nodes.end(), first_seed);
    if (it != hub_nodes.end()) used_hubs[distance(hub_nodes.begin(), it)] = true;
    
    if (procId == 0) cout << "Selected seed 1: " << first_seed << " with max distance " << max_dist << "\n" << endl;
    
    for (int k = 1; k < numParts; k++) {
        auto start_time = chrono::high_resolution_clock::now();

        int next_seed = find_farthest_hub(procId, nprocs, selected_seeds, hub_nodes, used_hubs, adj, seed_to_hub);

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

        if (next_seed == -1) {
            if (procId == 0) cout << "Warning: Could not find valid seed " << (k + 1) << endl;
            break;
        }

        selected_seeds.push_back(next_seed);

        auto it = find(hub_nodes.begin(), hub_nodes.end(), next_seed);
        if (it != hub_nodes.end()) {
            used_hubs[distance(hub_nodes.begin(), it)] = true;
        }

        if (procId == 0) {
            cout << "Selected seed " << (k + 1) << ": " << next_seed << " (took " << duration.count() << " ms)\n" << endl;
        }
    }

    result = selected_seeds;

    if (procId == 0) {
        cout << "Seed selection completed. Selected " << result.size() << " seeds: ";
        for (size_t i = 0; i < result.size(); i++) {
            cout << result[i];
            if (i < result.size() - 1) cout << ", ";
        }
        cout << endl;
    }

    return result;
}