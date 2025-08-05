#include <mpi.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <iterator>
#include <roaring/roaring.h>

#include "graph_types.h"
#include "utils.hpp"
#include "phase1/partition.hpp"

using namespace std;

void update_partition_csr(int partition_id, const vector<int> &partition_nodes, const unordered_map<int, vector<int>> &local_adj, Graph &partition_graph, GhostNodes &ghost_nodes) {
    partition_graph.clear();
    ghost_nodes.clear();

    if (partition_nodes.empty()) return;

    unordered_map<int, int> global_to_local;
    int local_idx = 0;
    for (int idx : partition_nodes) {
        global_to_local[idx] = local_idx++;
    }

    size_t partition_nodes_size = partition_nodes.size();
    partition_graph.num_vertices = static_cast<int>(partition_nodes_size);
    partition_graph.row_ptr.resize(partition_nodes_size + 1, 0);
    partition_graph.global_ids = partition_nodes;
    partition_graph.vertex_labels.resize(partition_nodes_size, partition_id);

    int edge_count = 0;
    for (size_t i = 0; i < partition_nodes_size; i++) {
        int gid = partition_nodes[i];
        auto adj_it = local_adj.find(gid);

        if (adj_it == local_adj.end()) {
            partition_graph.row_ptr[i + 1] = edge_count;
            continue;
        }

        const vector<int> &neighbors = adj_it->second;
        for (int ngid : neighbors) {
            if (global_to_local.count(ngid)) {
                partition_graph.col_indices.push_back(global_to_local[ngid]);
            } else {
                if (ghost_nodes.global_to_local.count(ngid) == 0) {
                    int ghost_idx = static_cast<int>(ghost_nodes.global_ids.size());
                    ghost_nodes.global_ids.push_back(ngid);
                    ghost_nodes.ghost_labels.push_back(-1);
                    ghost_nodes.global_to_local[ngid] = ghost_idx;
                }

                int ghost_idx = ghost_nodes.global_to_local[ngid];
                partition_graph.col_indices.push_back(partition_graph.num_vertices + ghost_idx);
            }

            edge_count++;
        }

        partition_graph.row_ptr[i + 1] = edge_count;
    }

    partition_graph.num_edges = edge_count;
}

int compute_partition_degree(roaring_bitmap_t *neighbors_bitmap, roaring_bitmap_t *partition_bitmap) {
    roaring_bitmap_t *intersection = roaring_bitmap_and(neighbors_bitmap, partition_bitmap);
    int count = static_cast<int>(roaring_bitmap_get_cardinality(intersection));
    roaring_bitmap_free(intersection);
    return count;
}

inline double compute_ratio(int node, roaring_bitmap_t *neighbors_bitmap, roaring_bitmap_t *partition_bitmap, const unordered_map<int, int> &global_degree) {
    auto it = global_degree.find(node);
    if (it == global_degree.end()) return 0.0;

    int node_degree = it->second;
    if (node_degree == 0) return 0.0;
    else if (node_degree == 1) return 1.0;

    int partition_degree = compute_partition_degree(neighbors_bitmap, partition_bitmap);

    return static_cast<double>(partition_degree) / node_degree;
}

vector<FrontierNode> collect_local_frontier_candidates(int procId, const vector<int> &partition_nodes, const unordered_map<int, roaring_bitmap_t*> &bitmap_adj, const unordered_set<int> &global_partitioned, const unordered_map<int, int> &global_degree, int partition_id, int theta) {
    if (partition_nodes.empty()) return vector<FrontierNode>();

    roaring_bitmap_t *partition_bitmap = create_partition_bitmap(partition_nodes);
    
    vector<pair<int, roaring_bitmap_t*>> adj_vector;
    adj_vector.reserve(bitmap_adj.size());
    for (const auto &pair : bitmap_adj) {
        adj_vector.push_back(pair);
    }

    vector<FrontierNode> local_candidates;
    size_t adj_vector_size = adj_vector.size();
    local_candidates.reserve(adj_vector_size);

    #pragma omp parallel
    {
        vector<FrontierNode> thread_candidates;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(adj_vector_size); ++i) {
            int node = adj_vector[i].first;
            roaring_bitmap_t *neighbors_bitmap = adj_vector[i].second;

            if (global_partitioned.count(node)) continue;

            int partition_degree = compute_partition_degree(neighbors_bitmap, partition_bitmap);
            if (partition_degree == 0) continue;

            double ratio = compute_ratio(node, neighbors_bitmap, partition_bitmap, global_degree);
            if (ratio <= 0.0) continue;

            auto degree_it = global_degree.find(node);
            if (degree_it == global_degree.end()) continue;

            int total_degree = degree_it->second;
            thread_candidates.emplace_back(node, ratio, partition_degree, total_degree, partition_id);
        }

        #pragma omp critical(merge_candidates)
        {
            local_candidates.insert(local_candidates.end(), thread_candidates.begin(), thread_candidates.end());
        }
    }

    roaring_bitmap_free(partition_bitmap);

    if (static_cast<int>(local_candidates.size()) > theta) {
        nth_element(local_candidates.begin(), local_candidates.begin() + theta, local_candidates.end(), greater<FrontierNode>());
        local_candidates.resize(theta);
    }

    sort(local_candidates.begin(), local_candidates.end(), greater<FrontierNode>());
    return local_candidates;
}

vector<FrontierNode> select_global_frontiers(int procId, int nprocs, const vector<FrontierNode> &local_frontiers, int theta) {
    vector<int> sendbuf;
    sendbuf.push_back(static_cast<int>(local_frontiers.size()));
    for (const auto &frontier : local_frontiers) {
        sendbuf.push_back(frontier.vertex);
        sendbuf.push_back(frontier.partition_degree);
        sendbuf.push_back(frontier.total_degree);
        sendbuf.push_back(static_cast<int>(frontier.ratio * 1000000));
        sendbuf.push_back(frontier.partition_id);
    }

    int send_size = static_cast<int>(sendbuf.size());

    vector<int> recv_counts(nprocs);
    MPI_Allgather(&send_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displs(nprocs);
    int total_size = 0;
    for (int p = 0; p < nprocs; p++) {
        displs[p] = total_size;
        total_size += recv_counts[p];
    }
    if (total_size == 0) return vector<FrontierNode>();

    vector<int> recvbuf(total_size);
    MPI_Allgatherv(sendbuf.data(), send_size, MPI_INT, recvbuf.data(), recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    vector<FrontierNode> all_frontiers;
    int idx = 0;
    for (int p = 0; p < nprocs; p++) {
        if (recv_counts[p] == 0) continue;

        int num_frontiers = recvbuf[idx++];
        for (int i = 0; i < num_frontiers; i++) {
            int vertex = recvbuf[idx++];
            int partition_degree = recvbuf[idx++];
            int total_degree = recvbuf[idx++];
            double ratio = static_cast<double>(recvbuf[idx++]) / 1000000.0;
            int partition_id = recvbuf[idx++];

            all_frontiers.emplace_back(vertex, ratio, partition_degree, total_degree, partition_id);
        }
    }

    unordered_map<int, FrontierNode> best_frontiers;
    for (const auto &frontier : all_frontiers) {
        auto it = best_frontiers.find(frontier.vertex);
        if (it == best_frontiers.end()) {
            best_frontiers[frontier.vertex] = frontier;
        } else {
            const double ratio_diff = frontier.ratio - it->second.ratio;
            const double epsilon = 1e-9;

            if (ratio_diff > epsilon) {
                it->second = frontier;
            } else if (abs(ratio_diff) <= epsilon) {
                if (frontier.partition_degree > it->second.partition_degree) {
                    it->second = frontier;
                } else if (frontier.partition_degree == it->second.partition_degree) {
                    if (frontier.partition_id < it->second.partition_id) it->second = frontier;
                }
            }
        }
    }

    vector<FrontierNode> global_frontiers;
    global_frontiers.reserve(best_frontiers.size());
    for (const auto &[node, frontier] : best_frontiers) {
        global_frontiers.push_back(frontier);
    }

    if (static_cast<int>(global_frontiers.size()) > theta) {
        nth_element(global_frontiers.begin(), global_frontiers.begin() + theta, global_frontiers.end(), greater<FrontierNode>());
        global_frontiers.resize(theta);
    }

    sort(global_frontiers.begin(), global_frontiers.end(), greater<FrontierNode>());
    return global_frontiers;
}

vector<PartitionUpdate> resolve_race_condition(int procId, int nprocs, const vector<FrontierNode> &candidates, unordered_set<int> &global_partitioned) {
    vector<PartitionUpdate> updates;
    vector<FrontierNode> sorted_candidates = candidates;
    sort(sorted_candidates.begin(), sorted_candidates.end(), [](const FrontierNode &a, const FrontierNode &b) {
        if (abs(a.ratio - b.ratio) > 1e-9) return a.ratio > b.ratio;
        if (a.partition_degree != b.partition_degree) return a.partition_degree > b.partition_degree;
        return a.vertex < b.vertex;
    });

    for (const auto &candidate : sorted_candidates) {
        if (global_partitioned.find(candidate.vertex) != global_partitioned.end()) continue;

        int local_decision = 1;
        int global_decision;
        MPI_Allreduce(&local_decision, &global_decision, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (global_decision == 1) updates.emplace_back(candidate.partition_id, candidate.vertex);
    }

    return updates;
}

int find_smallest_partition(const vector<vector<int>> &partitions) {
    size_t min_size = SIZE_MAX;
    int min_partition = 0;
    size_t size = partitions.size();

    for (size_t p = 0; p < size; p++) {
        size_t current_size = partitions[p].size();
        if (current_size < min_size) {
            min_size = current_size;
            min_partition = static_cast<int>(p);
        } else if (current_size == min_size && static_cast<int>(p) < min_partition) {
            min_partition = static_cast<int>(p);
        }
    }
    
    return min_partition;
}

vector<PartitionUpdate> handle_isolated_nodes(int procId, int nprocs, int numParts, const unordered_set<int> &all_nodes, const unordered_set<int> &global_partitioned, const vector<vector<int>> &partitions, int theta) {
    vector<PartitionUpdate> local_updates;
    vector<int> isolated_nodes;
    int node_index = 0;

    for (int node : all_nodes) {
        if (global_partitioned.find(node) == global_partitioned.end()) {
            if (node_index % nprocs == procId) isolated_nodes.push_back(node);
            node_index++;
        }
    }

    int max_batch = min(theta / numParts, static_cast<int>(isolated_nodes.size()));

    if (!isolated_nodes.empty() && max_batch > 0) {
        vector<pair<size_t, int>> partition_sizes;
        for (int p = 0; p < numParts; p++) {
            partition_sizes.push_back({partitions[p].size(), p});
        }
        sort(partition_sizes.begin(), partition_sizes.end());

        for (int i = 0; i < max_batch && i < static_cast<int>(isolated_nodes.size()); i++) {
            int target_partition = partition_sizes[i % numParts].second;
            local_updates.emplace_back(target_partition, isolated_nodes[i]);
        }
    }

    return local_updates;
}

void update_ghost_labels(int procId, int nprocs, vector<GhostNodes> &partition_ghosts, const vector<vector<int>> &partitions) {
    unordered_map<int, int> node_to_partition;
    int partitions_size = static_cast<int>(partitions.size());
    for (int p = 0; p < partitions_size; p++) {
        for (int node : partitions[p]) {
            node_to_partition[node] = p;
        }
    }

    vector<int> local_mapping;
    local_mapping.push_back(static_cast<int>(node_to_partition.size()));
    for (const auto &[node, partition] : node_to_partition) {
        local_mapping.push_back(node);
        local_mapping.push_back(partition);
    }

    int local_size = static_cast<int>(local_mapping.size());
    vector<int> recv_counts(nprocs);
    MPI_Allgather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displs(nprocs);
    int total_size = 0;
    for (int p = 0; p < nprocs; p++) {
        displs[p] = total_size;
        total_size += recv_counts[p];
    }

    if (total_size > 0) {
        vector<int> global_mapping(total_size);
        MPI_Allgatherv(local_mapping.data(), local_size, MPI_INT, global_mapping.data(), recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        unordered_map<int, int> global_node_to_partition;
        int idx = 0;
        for (int p = 0; p < nprocs; p++) {
            if (recv_counts[p] == 0) continue;

            int num_nodes = global_mapping[idx++];
            for (int i = 0; i < num_nodes; i++) {
                int node = global_mapping[idx++];
                int partition = global_mapping[idx++];
                global_node_to_partition[node] = partition;
            }
        }

        for (auto &ghost : partition_ghosts) {
            for (size_t i = 0; i < ghost.global_ids.size(); i++) {
                int ghost_node = ghost.global_ids[i];
                auto it = global_node_to_partition.find(ghost_node);
                if (it != global_node_to_partition.end()) ghost.ghost_labels[i] = it->second;
            }
        }
    }
}

void sync_csr_updates(int procId, int nprocs, const vector<PartitionUpdate> &updates, const unordered_map<int, vector<int>> &local_adj, vector<Graph> &partition_graphs, vector<GhostNodes> &partition_ghosts, const vector<vector<int>> &partitions) {
    unordered_set<int> updated_partitions;
    for (const auto &update : updates) {
        updated_partitions.insert(update.partition_id);
    }

    for (int id : updated_partitions) {
        update_partition_csr(id, partitions[id], local_adj, partition_graphs[id], partition_ghosts[id]);
    }

    update_ghost_labels(procId, nprocs, partition_ghosts, partitions);
}

void partition_expansion(int procId, int nprocs, int numParts, int theta, const vector<int> &seeds, const unordered_map<int, int> &global_degree, const unordered_map<int, vector<int>> &local_adj, vector<vector<int>> &partitions) {
    unordered_set<int> all_nodes;
    for (const auto &[node, degree] : global_degree) {
        all_nodes.insert(node);
    }

    unordered_set<int> global_partitioned;
    partitions.resize(numParts);
    unordered_map<int, roaring_bitmap_t*> bitmap_adj = convert_adj(local_adj);

    vector<Graph> partition_graphs(numParts);
    vector<GhostNodes> partition_ghosts(numParts);

    if (static_cast<int>(seeds.size()) != numParts) {
        cerr << "[Error] Total number of seeds must be same with number of partition\n";
        free_converted_graph(bitmap_adj);
        return;
    }

    vector<int> sorted_seeds = seeds;
    sort(sorted_seeds.begin(), sorted_seeds.end());

    for (int i = 0; i < numParts; i++) {
        partitions[i].push_back(sorted_seeds[i]);
        global_partitioned.insert(sorted_seeds[i]);

        update_partition_csr(i, partitions[i], local_adj, partition_graphs[i], partition_ghosts[i]);
    }

    update_ghost_labels(procId, nprocs, partition_ghosts, partitions);

    if (procId == 0) {
        cout << "[proc " << procId << "] Seed distribution\n";
        for (int p = 0; p < numParts; p++) {
            cout << " Partition " << p << ": Node" << sorted_seeds[p] << "\n";
        }
        cout << "[proc " << procId << "] Starting iterative expansion...\n";
    }

    int iteration = 0;
    const size_t total_nodes = all_nodes.size();

    while (global_partitioned.size() < total_nodes) {
        iteration++;

        if (procId == 0) cout << "[proc " << procId << "] Iteration " << iteration << " - Partitioned: " << global_partitioned.size() << "/" << all_nodes.size() << "\n";

        bool expansion = false;

        for (int p = 0; p < numParts; p++) {
            if (partitions[p].empty()) continue;

            vector<FrontierNode> local_frontiers = collect_local_frontier_candidates(procId, partitions[p], bitmap_adj, global_partitioned, global_degree, p, theta);
            vector<FrontierNode> global_frontiers = select_global_frontiers(procId, nprocs, local_frontiers, theta);
            vector<PartitionUpdate> updates = resolve_race_condition(procId, nprocs, global_frontiers, global_partitioned);

            if (!updates.empty()) {
                sync_updates(procId, nprocs, updates, partitions);
                sync_partitioned_status(procId, nprocs, updates, global_partitioned);
                sync_csr_updates(procId, nprocs, updates, local_adj, partition_graphs, partition_ghosts, partitions);
                
                expansion = true;

                if (procId == 0) {
                    cout << "[proc " << procId << "] Partition " << p << " added " << updates.size() << " nodes (new size: " << partitions[p].size() << ")\n";
                    cout << "[proc " << procId << "] CSR - Vertices: " << partition_graphs[p].num_vertices << ", Edges: " << partition_graphs[p].num_edges << ", Ghosts: " << partition_ghosts[p].global_ids.size() << "\n";
                    
                    if (!partition_ghosts[p].global_ids.empty()) {
                        cout << "[proc " << procId << "] Sample Ghost Labels for Partition " << p << ": ";
                        int sample_count = min(5, static_cast<int>(partition_ghosts[p].global_ids.size()));
                        for (int g = 0; g < sample_count; g++) {
                            cout << "Node" << partition_ghosts[p].global_ids[g] << "(label:" << partition_ghosts[p].ghost_labels[g] << ") ";
                        }
                        cout << "\n";
                    }
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (procId == 0) cout << "\n";
        
        if (!expansion) {
            size_t remaining_nodes = total_nodes - global_partitioned.size();
            if (remaining_nodes > 0) {
                vector<PartitionUpdate> local_isolated_updates = handle_isolated_nodes(procId, nprocs, numParts, all_nodes, global_partitioned, partitions, theta);
                vector<PartitionUpdate> all_isolated_updates = collect_updates(procId, nprocs, local_isolated_updates);
            
                if (!all_isolated_updates.empty()) {
                    vector<FrontierNode> isolated_frontiers;
                    for (const auto &update : all_isolated_updates) {
                        isolated_frontiers.emplace_back(update.node, 1.0, 1, 1, update.partition_id);
                    }

                    vector<PartitionUpdate> final_isolated_updates = resolve_race_condition(procId, nprocs, isolated_frontiers, global_partitioned);

                    if (!final_isolated_updates.empty()) {
                        sync_updates(procId, nprocs, final_isolated_updates, partitions);
                        sync_partitioned_status(procId, nprocs, final_isolated_updates, global_partitioned);
                        sync_csr_updates(procId, nprocs, final_isolated_updates, local_adj, partition_graphs, partition_ghosts, partitions);
                        
                        expansion = true;

                        if (procId == 0) cout << "[proc " << procId << "] Processed " << final_isolated_updates.size() << " isolated nodes\n";
                    }
                }

            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free_converted_graph(bitmap_adj);

    if (procId == 0) {
        cout << "[proc " << procId << "] Partition expansion completed after " << iteration << " iterations\n";
        cout << "[proc " << procId << "] Final CSR Statistics:\n";
        for (int p = 0; p < numParts; p++) {
            cout << "[proc " << procId << "] Partition " << p 
                 << " - Nodes: " << partitions[p].size()
                 << ", CSR Vertices: " << partition_graphs[p].num_vertices
                 << ", CSR Edges: " << partition_graphs[p].num_edges
                 << ", Ghost Nodes: " << partition_ghosts[p].global_ids.size() << "\n";
            
            if (!partition_ghosts[p].global_ids.empty()) {
                unordered_map<int, int> label_count;
                for (int label : partition_ghosts[p].ghost_labels) {
                    label_count[label]++;
                }
                cout << "[proc " << procId << "] Partition " << p << " Ghost Label Distribution: ";
                for (const auto &[label, count] : label_count) {
                    cout << "P" << label << ":" << count << " ";
                }
                cout << "\n";
            }
        }
    }
}