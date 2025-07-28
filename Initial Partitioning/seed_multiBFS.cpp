#include <mpi.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <set>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <iomanip>

#include "bfs.hpp"
#include "seed.hpp"

using namespace std;

const int INF = INT_MAX;

vector<unordered_map<int, int>> compute_distances(int procId, int nprocs, const vector<int> &landmarks, const vector<int> &hub_nodes, const unordered_map<int, vector<int>> &adj) {
    int L = landmarks.size();
    int H = hub_nodes.size();
    if (L == 0 || H == 0) return {};

    int landmarks_per_proc = (L + nprocs - 1) / nprocs;
    int start_idx = procId * landmarks_per_proc;
    int end_idx = min(L, start_idx + landmarks_per_proc);
    int local_count = max(0, end_idx - start_idx);

    unordered_map<int, int> hub_idx;
    for (int h = 0; h < H; ++h)
        hub_idx[hub_nodes[h]] = h;

    vector<unordered_map<int, int>> distances(L);
    vector<vector<int>> local_results(local_count, vector<int>(H, INF));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < local_count; ++i) {
        int landmark_idx = start_idx + i;
        int landmark = landmarks[landmark_idx];

        auto bfs_result = batch_bfs(landmark, procId, nprocs, hub_nodes, adj);

        for (const auto &[node, dist] : bfs_result) {
            if (hub_idx.count(node)) {
                int hub_i = hub_idx[node];
                local_results[i][hub_i] = dist;
            }
        }
    }

    vector<int> sendbuf(L * H, INF);
    for (int i = 0; i < local_count; ++i) {
        for (int h = 0; h < H; ++h) {
            int global_idx = start_idx + 1;
            sendbuf[global_idx * H + h] = local_results[i][h];
        }
    }

    vector<int> recvbuf(L * H, INF);
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), L * H, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h)
            if (recvbuf[l * H + h] != INF) distances[l][hub_nodes[h]] = recvbuf[l * H + h];
    }

    if (procId == 0) cout << "compute_distances completed with batch BFS" << endl;

    return distances;
}

vector<int> compute_hub_pairs_distances(int procId, int nprocs, const vector<int> &landmarks, const vector<int> &hub_nodes, const unordered_map<int, vector<int>> &adj) {
    vector<unordered_map<int, int>> distances = compute_distances(procId, nprocs, landmarks, hub_nodes, adj);
    int L = landmarks.size();
    int H = hub_nodes.size();
    if (L == 0 || H == 0) return {};

    unordered_map<int, int> hub_idx;
    for (int h = 0; h < H; ++h)
        hub_idx[hub_nodes[h]] = h;

    vector<vector<int>> landmarks_distances(L, vector<int>(H, INF));
    for (int l = 0; l < L; ++l) {
        for (const auto &[hub, d] : distances[l])
            if (hub_idx.count(hub)) landmarks_distances[l][hub_idx[hub]] = d;
    }

    int vector_size = H * (H - 1) / 2;
    vector<int> local(vector_size, INF);
    vector<int> result(vector_size, INF);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < H; ++i) {
        for (int j = i + 1; j < H; ++j) {
            int min_dist = INF;

            for (int l = 0; l < L; ++l) {
                int d1 = landmarks_distances[l][i];
                int d2 = landmarks_distances[l][j];
                if (d1 != INF && d2 != INF) min_dist = min(min_dist, d1 + d2);
            }

            int idx = i * H - (i + 1) * (i + 2) / 2 + j;
            local[idx] = min_dist;
        }
    }

    MPI_Allreduce(local.data(), result.data(), vector_size, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (procId == 0) cout << "compute_hub_pairs_distances completed" << "\n";

    return result;
}

int hub_distance_indexed(int i, int j, int H, const vector<int> &distances) {
    if (i == j) return 0;
    
    int idx;
    if (i < j) idx = i * H - (i + 1) * (i + 2) / 2 + j;
    else idx = j * H - (j + 1) * (j + 2) / 2 + i;

    if (idx < 0 || idx >= distances.size()) return INF;

    return distances[idx];
}

vector<unordered_map<int, int>> compute_seeds_to_hubs_distances(int procId, int nprocs, const vector<int> &selected_seeds, const vector<int> &hub_nodes, const unordered_map<int, vector<int>> &adj) {
    int S = selected_seeds.size();
    int H = hub_nodes.size();
    if (S == 0 || H == 0) return {};

    int seeds_per_proc = (S + nprocs - 1) / nprocs;
    int start_idx = procId * seeds_per_proc;
    int end_idx = min(S, start_idx + seeds_per_proc);
    int local_count = max(0, end_idx - start_idx);

    unordered_map<int, int> hub_idx;
    for (int h = 0; h < H; ++h)
        hub_idx[hub_nodes[h]] = h;

    vector<unordered_map<int, int>> distances(S);
    vector<vector<int>> local_results(local_count, vector<int>(H, INF));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < local_count; ++i) {
        int seed_idx = start_idx + i;
        int seed = selected_seeds[seed_idx];

        auto bfs_result = batch_bfs(seed, procId, nprocs, hub_nodes, adj);

        for (const auto &[node, dist] : bfs_result) {
            if (hub_idx.count(node)) {
                int hub_i = hub_idx[node];
                local_results[i][hub_i] = dist;
            }
        }
    }

    vector<int> sendbuf(S * H, INF);
    for (int i = 0; i < local_count; ++i) {
        for (int h = 0; h < H; ++h) {
            int global_idx = start_idx + i;
            sendbuf[global_idx * H + h] = local_results[i][h];
        }
    }

    vector<int> recvbuf(S * H, INF);
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), S * H, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < H; ++h)
            if (recvbuf[s * H + h] != INF) distances[s][hub_nodes[h]] = recvbuf[s * H + h];
    }

    return distances;
}

vector<int> find_seeds(int procId, int nprocs, int numParts, const vector<int> &landmarks, const vector<int> &hub_nodes, const unordered_map<int, vector<int>> &adj) {
    vector<int> hub_distances = compute_hub_pairs_distances(procId, nprocs, landmarks, hub_nodes, adj);
    int H = hub_nodes.size();

    if (numParts > H) {
        if (procId == 0)  cout << "Warning: numParts (" << numParts << ") > number of hub nodes (" << H << ")" << endl;
        return {};
    } else if (numParts == 0) return {};

    vector<int> selected_seeds;
    vector<bool> used(H, false);

    int max_dist = -1;
    int best_i = -1, best_j = -1;

    for (int i = 0; i < H; ++i) {
        for (int j = i + 1; j < H; ++j) {
            int dist = hub_distance_indexed(i, j, H, hub_distances);
            if (dist != INF && dist > max_dist) {
                max_dist = dist;
                best_i = i;
                best_j = j;
            }
        }
    }

    if (best_i == -1 || best_j == -1) {
        if (procId == 0) cout << "Error: No valid hub pair found" << endl;
        return {};
    }

    selected_seeds.push_back(hub_nodes[best_i]);
    selected_seeds.push_back(hub_nodes[best_j]);
    used[best_i] = true;
    used[best_j] = true;

    if (procId == 0) cout << "Initial pair: hub " << best_i << " (node " << hub_nodes[best_i] << ") and hub " << best_j << " (node " << hub_nodes[best_j] << ") with distance " << max_dist << endl;

    for (int k = 2; k < numParts; ++k) {
        auto seed_distances = compute_seeds_to_hubs_distances(procId, nprocs, selected_seeds, hub_nodes, adj);

        if (seed_distances.empty()) {
            if (procId == 0) cout << "Error: Failed to compute distances from seeds" << endl;
            break;
        }

        int best_hub_idx = -1;
        long long max_total_dist = -1;

        #pragma omp parallel for schedule(dynamic)
        for (int h = 0; h < H; ++h) {
            if (used[h]) continue;

            int hub_node = hub_nodes[h];
            long long total_dist = 0;
            bool valid = true;

            for (int s = 0; s < selected_seeds.size(); ++s) {
                auto it = seed_distances[s].find(hub_node);
                if (it == seed_distances[s].end() || it->second == INF) {
                    valid = false;
                    break;
                }
                total_dist += it->second;
            }

            if (valid) {
                #pragma omp critical
                {
                    if (total_dist > max_total_dist) {
                        max_total_dist = total_dist;
                        best_hub_idx = h;
                    }
                }
            }
        }

        if (best_hub_idx == -1) {
            if (procId == 0) cout << "Warning: No valid candidate found at step " << k << endl;
            break;
        }

        selected_seeds.push_back(hub_nodes[best_hub_idx]);
        used[best_hub_idx] = true;
    }

    if (procId == 0) {
        cout << "find_seeds completed with " << selected_seeds.size() << " seeds found using multi-BFS" << endl;
        cout << "Selected seed nodes: ";
        for (int i = 0; i < selected_seeds.size(); ++i) {
            cout << selected_seeds[i];
            if (i < selected_seeds.size() - 1) cout << ", ";
        }
        cout << endl;
    }
    
    return selected_seeds;
}