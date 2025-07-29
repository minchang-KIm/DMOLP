#include <mpi.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>

#include "bfs.hpp"

using namespace std;

const int INF = INT_MAX;
const int MAX_HOPS = 6;
const int BATCH_SIZE = 10000;

unordered_map<int, int> batch_bfs(int source, int procId, int nprocs, const vector<int> &hub_nodes, const unordered_map<int, vector<int>> &local_adj) {
    unordered_map<int, int> distances;
    set<int> hub_set(hub_nodes.begin(), hub_nodes.end());

    vector<int> current;
    vector<int> next;
    unordered_set<int> visited;

    current.push_back(source);
    distances[source] = 0;
    visited.insert(source);

    int hop = 0;
    int found_hubs = 0;
    int H = hub_nodes.size();

    while (!current.empty() && hop < MAX_HOPS && found_hubs < H) {
        next.clear();
        const size_t current_size = current.size();

        for (size_t batch_start = 0; batch_start < current_size; batch_start += BATCH_SIZE) {
            size_t batch_end = min(current_size, batch_start + BATCH_SIZE);

            vector<int> local_neighbors;
            
            #pragma omp parallel
            {
                vector<int> thread_neighbors;

                #pragma omp for
                for (size_t i = batch_start; i < batch_end; ++i) {
                    int current_node = current[i];

                    if (hub_set.count(current_node)) {
                        #pragma omp critical
                        {
                            if (distances.find(current_node) == distances.end()) {
                                distances[current_node] = hop;
                                found_hubs++;
                            }
                        }
                    }

                    if (current_node % nprocs == procId && local_adj.count(current_node)) {
                        for (int neighbor : local_adj.at(current_node)) {
                            if (visited.find(neighbor) == visited.end()) thread_neighbors.push_back(neighbor);
                        }
                    }
                }

                #pragma omp critical
                {
                    local_neighbors.insert(local_neighbors.end(), thread_neighbors.begin(), thread_neighbors.end());
                }
            }

            sort(local_neighbors.begin(), local_neighbors.end());
            local_neighbors.erase(unique(local_neighbors.begin(), local_neighbors.end()), local_neighbors.end());

            vector<int> send_counts(nprocs, 0);
            vector<int> recv_counts(nprocs);
            vector<int> send_displs(nprocs, 0);
            vector<int> recv_displs(nprocs);

            send_counts[procId] = local_neighbors.size();
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            int total_neighbors = 0;
            for (int p = 0; p < nprocs; ++p) {
                recv_counts[p] = send_counts[p];
                recv_displs[p] = total_neighbors;
                total_neighbors += recv_counts[p];
            }

            vector<int> all_neighbors(total_neighbors);
            MPI_Allgatherv(local_neighbors.data(), local_neighbors.size(), MPI_INT, all_neighbors.data(), recv_counts.data(), recv_displs.data(), MPI_INT, MPI_COMM_WORLD);

            for (int neighbor : all_neighbors) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    next.push_back(neighbor);
                }
            }
        }

        current = move(next);
        hop++;

        int global_found_hubs;
        MPI_Allreduce(&found_hubs, &global_found_hubs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        found_hubs = global_found_hubs;
    }

    for (int hub : hub_nodes) {
        if (distances.find(hub) == distances.end()) distances[hub] = INF;
    }

    return distances;
}