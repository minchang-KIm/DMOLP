#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdint>

#include "graph_types.h"
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

    if (total_size == 0) return;

    vector<int> total_serialized(total_size);
    MPI_Allgatherv(local_serialized.data(), local_size, MPI_INT, total_serialized.data(), all_sizes.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    vector<PartitionUpdate> all_updates;

    for (int p = 0; p < nprocs; p++) {
        if (all_sizes[p] == 0) continue;

        vector<int> proc_data(total_serialized.begin() + displacements[p], total_serialized.begin() + displacements[p] + all_sizes[p]);
        vector<PartitionUpdate> proc_updates = deserialize_updates(proc_data);
        all_updates.insert(all_updates.end(), proc_updates.begin(), proc_updates.end());
    }

    sort(all_updates.begin(), all_updates.end(), [](const PartitionUpdate &a, const PartitionUpdate &b) {
        if (a.node != b.node) return a.node < b.node;
        return a.partition_id < b.partition_id;
    });

    unordered_set<int> processed_nodes;
    for (const auto &u : all_updates) {
        if (processed_nodes.count(u.node)) continue;

        if (global_partitioned.find(u.node) == global_partitioned.end()) {
            partitions[u.partition_id].push_back(u.node);
            global_partitioned.insert(u.node);
            processed_nodes.insert(u.node);
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
        partitions[i] = vector<int>(buffer.begin() + idx, buffer.begin() + idx + size);
        idx += size;
    }

    return partitions;
}

void sync_global_partitions(int procId, int nprocs, vector<vector<int>> &partitions, unordered_set<int> &global_partitioned, vector<PartitionUpdate> &pending_updates) {
    if (pending_updates.empty()) return;

    sync_updates(procId, nprocs, pending_updates, partitions, global_partitioned);

    pending_updates.clear();
}

void sync_partitioned_status(int procId, int nprocs, unordered_set<int> &global_partitioned) {
    vector<int> local_partitioned(global_partitioned.begin(), global_partitioned.end());
    int local_size = (int)local_partitioned.size();

    vector<int> all_sizes(nprocs);
    MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displacements(nprocs);
    int total_size = 0;
    for (int p = 0; p < nprocs; p++) {
        displacements[p] = total_size;
        total_size += all_sizes[p];
    }

    if (total_size == 0) return;

    vector<int> all_partitioned(total_size);
    MPI_Allgatherv(local_partitioned.data(), local_size, MPI_INT, all_partitioned.data(), all_sizes.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    global_partitioned.clear();
    global_partitioned.insert(all_partitioned.begin(), all_partitioned.end());
}

void sync_partitioned(int procId, int nprocs, const vector<int> &newly_partitioned, unordered_set<int> &global_partitioned) {
    int local_size = (int)newly_partitioned.size();

    vector<int> all_sizes(nprocs);
    MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displacements(nprocs);
    int total_size = 0;
    for (int p = 0; p < nprocs; p++) {
        displacements[p] = total_size;
        total_size += all_sizes[p];
    }

    if (total_size == 0) return;

    vector<int> all_partitioned(total_size);
    MPI_Allgatherv(newly_partitioned.data(), local_size, MPI_INT, all_partitioned.data(), all_sizes.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    global_partitioned.insert(all_partitioned.begin(), all_partitioned.end());
}

void add_node_to_partition(int node, int partition_id, vector<vector<int>> &partitions, unordered_set<int> &global_partitioned, vector<PartitionUpdate> &pending_updates) {
    if (global_partitioned.find(node) == global_partitioned.end()) {
        partitions[partition_id].push_back(node);
        global_partitioned.insert(node);
        pending_updates.emplace_back(partition_id, node);
    }
}

void seed_redistribution(int procId, int nprocs, int numParts, const vector<int> &remaining_seeds, const unordered_map<int, vector<int>> &local_adj, vector<vector<int>> &partitions, unordered_set<int> &global_partitioned) {
    vector<int> all_empty(nprocs, 0);
    int is_empty = remaining_seeds.empty() ? 1 : 0;
    MPI_Allgather(&is_empty, 1, MPI_INT, all_empty.data(), 1, MPI_INT, MPI_COMM_WORLD);

    bool empty = true;
    for (int flag : all_empty) {
        if (flag == 0) {
            empty = false;
            break;
        }
    }
    if (empty) return;

    vector<NodeInfo> remaining_seed_infos;
    remaining_seed_infos.reserve(remaining_seeds.size());
    for (int seed : remaining_seeds) {
        auto it = local_adj.find(seed);
        vector<int> neighbors = (it != local_adj.end()) ? it->second : vector<int>();
        remaining_seed_infos.push_back({seed, neighbors});
    }

    vector<int> sendbuf = serialize_node_info(remaining_seed_infos);
    int local_size = (int)sendbuf.size();

    vector<int> all_sizes(nprocs);
    MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displacements(nprocs);
    int total_size = 0;
    for (int p = 0; p < nprocs; p++) {
        displacements[p] = total_size;
        total_size += all_sizes[p];
    }

    if (total_size <= 0) {
        if (procId == 0) cout << "[proc " << procId << "] Warning: No data to redistribute\n";
        return;
    }

    vector<int> all_data;
    try {
        all_data.resize(total_size, 0);
    } catch (const bad_alloc &e) {
        cerr << "[proc " << procId << "] Memory allocation failed for size " << total_size << endl;
        return;
    }
    
    MPI_Allgatherv(sendbuf.data(), local_size, MPI_INT, all_data.data(), all_sizes.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);
    
    vector<NodeInfo> all_remaining_seeds;
    unordered_set<int> duplicated_seeds;

    for (int p = 0; p < nprocs; p++) {
        if (all_sizes[p] == 0) continue;

        vector<int> proc_data(all_data.begin() + displacements[p], all_data.begin() + displacements[p] + all_sizes[p]);
        vector<NodeInfo> proc_seeds = deserialize_node_info(proc_data);

        for (const auto &s : proc_seeds) {
            if (duplicated_seeds.find(s.vertex) == duplicated_seeds.end()) {
                duplicated_seeds.insert(s.vertex);
                all_remaining_seeds.push_back(s);
            }
        }
    }

    vector<PartitionUpdate> updates;
    vector<int> empty_partitions;

    for (int p = 0; p < numParts; p++)
        if (partitions[p].empty()) empty_partitions.push_back(p);

    sort(all_remaining_seeds.begin(), all_remaining_seeds.end(), [](const NodeInfo &a, const NodeInfo &b) {
        return a.vertex < b.vertex;
    });

    size_t seed_idx = 0;
    size_t remaining_size = all_remaining_seeds.size();
    for (int partition_id : empty_partitions) {
        if (seed_idx >= remaining_size) break;

        int seed = all_remaining_seeds[seed_idx].vertex;

        if (global_partitioned.find(seed) == global_partitioned.end()) {
            partitions[partition_id].push_back(seed);
            global_partitioned.insert(seed);
            updates.push_back(PartitionUpdate{partition_id, seed});
        }

        ++seed_idx;
    }

    while (seed_idx < remaining_size) {
        int min_partition = 0;
        size_t min_size = partitions[0].size();

        for (int p = 1; p < numParts; p++) {
            size_t current_size = partitions[p].size();
            if (current_size < min_size || (current_size == min_size && p < min_partition)) {
                min_size = current_size;
                min_partition = p;
            }
        }

        int seed = all_remaining_seeds[seed_idx].vertex;
        if (global_partitioned.find(seed) == global_partitioned.end()) {
            partitions[min_partition].push_back(seed);
            global_partitioned.insert(seed);
            updates.push_back(PartitionUpdate{min_partition, seed});
        }

        ++seed_idx;
    }

    if (procId == 0) {
        cout << "[proc " << procId << "] Seed redistribution summary:\n";
        cout << "  Total remaining seeds collected: " << all_remaining_seeds.size() << "\n";
        cout << "  Empty partitions found: " << empty_partitions.size() << "\n";
        cout << "  Updates generated: " << updates.size() << "\n";
        
        unordered_set<int> assigned_seeds;
        for (int p = 0; p < numParts; p++) {
            for (int node : partitions[p]) {
                if (assigned_seeds.count(node)) cout << "  WARNING: Node " << node << " assigned to multiple partitions!\n";
                else assigned_seeds.insert(node);
            }
        }
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

void print_proc_partition(int procId, int nprocs, int numParts, const vector<vector<int>> &partitions, const unordered_map<int, vector<int>> &local_adj) {
    vector<int> local(numParts, 0);

    for (int p = 0; p < numParts; p++) {
        for (int node : partitions[p])
            if (local_adj.find(node) != local_adj.end()) local[p]++;
    }

    vector<int> all_data(nprocs * numParts);
    MPI_Allgather(local.data(), numParts, MPI_INT, all_data.data(), numParts, MPI_INT, MPI_COMM_WORLD);

    if (procId == 0) {
        cout << "\n=== Process-Partition Matrix ===\n";
        cout << "Proc\\Partition";
        for (int p = 0; p < numParts; p++) {
            cout << "\tPart" << p;
        }
        cout << "\tTotal\n";
        
        for (int proc = 0; proc < nprocs; proc++) {
            cout << "Process " << proc;
            int proc_total = 0;
            
            for (int p = 0; p < numParts; p++) {
                int responsibility = all_data[proc * numParts + p];
                cout << "\t" << responsibility;
                proc_total += responsibility;
            }
            cout << "\t" << proc_total << "\n";
        }
        
        cout << "\n=== Process Workload Analysis ===\n";
        for (int proc = 0; proc < nprocs; proc++) {
            int proc_total = 0;
            vector<int> proc_partitions;
            
            for (int p = 0; p < numParts; p++) {
                int responsibility = all_data[proc * numParts + p];
                proc_total += responsibility;
                if (responsibility > 0) {
                    proc_partitions.push_back(p);
                }
            }
            
            cout << "Process " << proc << ": " << proc_total << " nodes, responsible for partitions: ";
            for (size_t i = 0; i < proc_partitions.size(); i++) {
                cout << proc_partitions[i];
                if (i < proc_partitions.size() - 1) cout << ", ";
            }
            cout << "\n";
        }
        
        vector<int> proc_loads(nprocs, 0);
        for (int proc = 0; proc < nprocs; proc++) {
            for (int p = 0; p < numParts; p++) {
                proc_loads[proc] += all_data[proc * numParts + p];
            }
        }
        
        int min_load = *min_element(proc_loads.begin(), proc_loads.end());
        int max_load = *max_element(proc_loads.begin(), proc_loads.end());
        double avg_load = 0.0;
        for (int load : proc_loads) avg_load += load;
        avg_load /= nprocs;
        
        cout << "\n=== Load Balance Metrics ===\n";
        cout << "Min process load: " << min_load << " nodes\n";
        cout << "Max process load: " << max_load << " nodes\n";
        cout << "Average process load: " << avg_load << " nodes\n";
        if (avg_load > 0) {
            cout << "Load imbalance ratio: " << (double)max_load / avg_load << "\n";
        }
        cout << "=======================================\n\n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void printComparisonReport(const PartitioningMetrics& m1, const PartitioningMetrics& m2) {
    std::cout << "\n┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    메트릭 비교 결과                         │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Edge-cut:                                                   │\n";
    std::cout << "│   Phase 1 (초기): " << std::setw(10) << m1.edge_cut << "                              │\n";
    std::cout << "│   Phase 2 (최종): " << std::setw(10) << m2.edge_cut << "                              │\n";
    double edge_cut_improvement = (m1.edge_cut > 0) ? (static_cast<double>(m1.edge_cut - m2.edge_cut) / m1.edge_cut) * 100.0 : 0.0;
    std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_cut_improvement << "%                             │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│ Vertex Balance:                                             │\n";
    std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << m1.vertex_balance << "                             │\n";
    std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << m2.vertex_balance << "                             │\n";
    double vertex_balance_improvement = (m1.vertex_balance > 0) ? ((m1.vertex_balance - m2.vertex_balance) / m1.vertex_balance) * 100.0 : 0.0;
    std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << vertex_balance_improvement << "%                             │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│ Edge Balance:                                               │\n";
    std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << m1.edge_balance << "                             │\n";
    std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << m2.edge_balance << "                             │\n";
    double edge_balance_improvement = (m1.edge_balance > 0) ? ((m1.edge_balance - m2.edge_balance) / m1.edge_balance) * 100.0 : 0.0;
    std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_balance_improvement << "%                             │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│ 실행 시간:                                                  │\n";
    std::cout << "│   Phase 1 (로딩): " << std::setw(6) << m1.loading_time_ms << " ms                          │\n";
    std::cout << "│   Phase 1 (분산): " << std::setw(6) << m1.distribution_time_ms << " ms                          │\n";
    std::cout << "│   Phase 2 (7단계): " << std::setw(5) << m2.loading_time_ms << " ms                          │\n";
    std::cout << "│   총 소요시간:     " << std::setw(5) << (m1.loading_time_ms + m1.distribution_time_ms + m2.loading_time_ms) << " ms                          │\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n";
    std::cout << "\n=== 알고리즘 성능 요약 ===\n";
    if (edge_cut_improvement > 0) {
        std::cout << "✓ Edge-cut " << std::fixed << std::setprecision(1) << edge_cut_improvement << "% 개선 ("
                  << m1.edge_cut << " → " << m2.edge_cut << ")\n";
    } else {
        std::cout << "⚠ Edge-cut " << std::fixed << std::setprecision(1) << -edge_cut_improvement << "% 악화 ("
                  << m1.edge_cut << " → " << m2.edge_cut << ")\n";
    }
    if (vertex_balance_improvement > 0) {
        std::cout << "✓ Vertex Balance " << std::fixed << std::setprecision(1) << vertex_balance_improvement << "% 개선 ("
                  << m1.vertex_balance << " → " << m2.vertex_balance << ")\n";
    } else {
        std::cout << "⚠ Vertex Balance " << std::fixed << std::setprecision(1) << -vertex_balance_improvement << "% 악화 ("
                  << m1.vertex_balance << " → " << m2.vertex_balance << ")\n";
    }
    if (edge_balance_improvement > 0) {
        std::cout << "✓ Edge Balance " << std::fixed << std::setprecision(1) << edge_balance_improvement << "% 개선 ("
                  << m1.edge_balance << " → " << m2.edge_balance << ")\n";
    } else {
        std::cout << "⚠ Edge Balance " << std::fixed << std::setprecision(1) << -edge_balance_improvement << "% 악화 ("
                  << m1.edge_balance << " → " << m2.edge_balance << ")\n";
    }
    std::cout << "총 소요시간: " << (m1.loading_time_ms + m1.distribution_time_ms + m2.loading_time_ms) << " ms\n";
    std::cout << "\n=== 7단계 알고리즘 완료 ===\n";
}