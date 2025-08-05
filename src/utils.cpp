#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <cstdint>
#include <cmath>
#include <roaring/roaring.h>

#include "utils.hpp"

using namespace std;

bool is_local_node(int node, int procId, int nprocs) {
    if (nprocs <= 1) return true;
    return (node % nprocs) == procId;
}

roaring_bitmap_t* create_partition_bitmap(const vector<int> &partition_nodes) {
    roaring_bitmap_t *bitmap = roaring_bitmap_create();
    for (int node : partition_nodes) {
        roaring_bitmap_add(bitmap, node);
    }
    return bitmap;
}

unordered_map<int, roaring_bitmap_t*> convert_adj(const unordered_map<int, vector<int>> &local_adj) {
    unordered_map<int, roaring_bitmap_t*> converted_result;
    vector<int> keys;
    vector<vector<int>> values;
    size_t adj_size = local_adj.size();

    keys.reserve(adj_size);
    values.reserve(adj_size);

    for (const auto &[node, neighbors] : local_adj) {
        keys.push_back(node);
        values.push_back(neighbors);
    }

    size_t keys_size = keys.size();
    vector<pair<int, roaring_bitmap_t*>> temp(keys_size);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(keys_size); i++) {
        int node = keys[i];
        const vector<int> &neighbors = values[i];

        roaring_bitmap_t *bitmap = roaring_bitmap_create();
        for (int neighbor : neighbors) {
            roaring_bitmap_add(bitmap, neighbor);
        }

        temp[i] = {node, bitmap};
    }

    for (const auto &[node, bitmap] : temp) {
        converted_result[node] = bitmap;
    }

    return converted_result;
}

void free_converted_graph(unordered_map<int, roaring_bitmap_t*> &bitmap_map) {
    for (auto &[node, bitmap] : bitmap_map) {
        if (bitmap != nullptr) {
            roaring_bitmap_free(bitmap);
            bitmap = nullptr;
        }
    }
    bitmap_map.clear();
}

roaring_bitmap_t* create_hub_bitmap(const vector<int> &hub_nodes) {
    roaring_bitmap_t *hub_bitmap = roaring_bitmap_create();
    for (int hub : hub_nodes) {
        roaring_bitmap_add(hub_bitmap, static_cast<uint32_t>(hub));
    }
    return hub_bitmap;
}

void broadcast_roaring_bitmap(roaring_bitmap_t *bitmap, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    size_t serialized_size = 0;
    char *serialized_data = nullptr;

    if (rank == root) {
        serialized_size = roaring_bitmap_portable_size_in_bytes(bitmap);
        serialized_data = new char[serialized_size];
        roaring_bitmap_portable_serialize(bitmap, serialized_data);
    }

    MPI_Bcast(&serialized_size, sizeof(size_t), MPI_BYTE, root, comm);

    if (rank != root) serialized_data = new char[serialized_size];

    MPI_Bcast(serialized_data, serialized_size, MPI_BYTE, root, comm);

    if (rank != root) {
        roaring_bitmap_clear(bitmap);
        roaring_bitmap_t *temp = roaring_bitmap_portable_deserialize(serialized_data);
        roaring_bitmap_or_inplace(bitmap, temp);
        roaring_bitmap_free(temp);
    }

    delete[] serialized_data;
}

void allreduce_roaring_bitmap_or(roaring_bitmap_t *local_bitmap, roaring_bitmap_t *result_bitmap, MPI_Comm comm) {
    if (roaring_bitmap_is_empty(local_bitmap)) {
        roaring_bitmap_clear(result_bitmap);
        return;
    }
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t local_size = roaring_bitmap_portable_size_in_bytes(local_bitmap);
    char *local_data = new char[local_size];
    roaring_bitmap_portable_serialize(local_bitmap, local_data);

    vector<size_t> all_sizes(size);
    MPI_Allgather(&local_size, sizeof(size_t), MPI_BYTE, all_sizes.data(), sizeof(size_t), MPI_BYTE, comm);

    vector<int> displs(size);
    size_t total_size = 0;
    for (int i = 0; i < size; i++) {
        displs[i] = static_cast<int>(total_size);
        total_size += all_sizes[i];
    }

    char *all_data = new char[total_size];
    vector<int> int_sizes(size);
    for (int i = 0; i < size; i++) {
        int_sizes[i] = static_cast<int>(all_sizes[i]);
    }

    MPI_Allgatherv(local_data, static_cast<int>(local_size), MPI_BYTE, all_data, int_sizes.data(), displs.data(), MPI_BYTE, comm);
    roaring_bitmap_clear(result_bitmap);

    size_t offset = 0;
    for (int i = 0; i < size; i++) {
        if (all_sizes[i] > 0) {
            roaring_bitmap_t *temp = roaring_bitmap_portable_deserialize(all_data + offset);
            roaring_bitmap_or_inplace(result_bitmap, temp);
            roaring_bitmap_free(temp);
            offset += all_sizes[i];
        }
    }

    delete[] local_data;
    delete[] all_data;
}

void sync_vector(int procId, int sourceProc, vector<int> &vec) {
    int size = static_cast<int>(vec.size());
    MPI_Bcast(&size, 1, MPI_INT, sourceProc, MPI_COMM_WORLD);

    if (procId != sourceProc && size > 0) vec.resize(size);

    if (size > 0) MPI_Bcast(vec.data(), size, MPI_INT, sourceProc, MPI_COMM_WORLD);
}

vector<int> serialize_updates(const vector<PartitionUpdate> &updates) {
    vector<int> buffer;
    buffer.push_back(static_cast<int>(updates.size()));

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

vector<PartitionUpdate> collect_updates(int procId, int nprocs, const vector<PartitionUpdate> &local_updates) {
    vector<int> serialized = serialize_updates(local_updates);
    int local_size = static_cast<int>(serialized.size());

    vector<int> all_sizes(nprocs);
    MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displs(nprocs);
    int total_size = 0;
    for (int p = 0; p < nprocs; p++) {
        displs[p] = total_size;
        total_size += all_sizes[p];
    }

    vector<PartitionUpdate> all_updates;

    if (total_size > 0) {
        vector<int> all_serialized(total_size);
        MPI_Allgatherv(serialized.data(), local_size, MPI_INT, all_serialized.data(), all_sizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        int idx = 0;
        for (int p = 0; p < nprocs; p++) {
            if (all_sizes[p] > 0) {
                vector<int> proc_data(all_serialized.begin() + idx, all_serialized.begin() + idx + all_sizes[p]);
                vector<PartitionUpdate> proc_updates = deserialize_updates(proc_data);
                all_updates.insert(all_updates.end(), proc_updates.begin(), proc_updates.end());
                idx += all_sizes[p];
            }
        }
    }

    return all_updates;
}

void apply_updates(const vector<PartitionUpdate> &updates, vector<vector<int>> &partitions) {
    for (const auto &update : updates) {
        if (update.partition_id >= 0 && update.partition_id < static_cast<int>(partitions.size())) partitions[update.partition_id].push_back(update.node);
    }
}

void sync_updates(int procId, int nprocs, const vector<PartitionUpdate> &updates, vector<vector<int>> &partitions) {
    vector<int> serialized_updates;
    int update_size = 0;

    if (procId == 0) {
        serialized_updates = serialize_updates(updates);
        update_size = static_cast<int>(serialized_updates.size());
    }

    MPI_Bcast(&update_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (update_size > 0) {
        if (procId != 0) serialized_updates.resize(update_size);
        MPI_Bcast(serialized_updates.data(), update_size, MPI_INT, 0, MPI_COMM_WORLD);

        vector<PartitionUpdate> received_updates = deserialize_updates(serialized_updates);
        apply_updates(received_updates, partitions);
    }
}

void sync_partitioned_status(int procId, int nprocs, const vector<PartitionUpdate> &updates, unordered_set<int> &global_partitioned) {
    vector<int> local_new_nodes;
    for (const auto &update : updates) {
        local_new_nodes.push_back(update.node);
    }

    int local_count = static_cast<int>(local_new_nodes.size());

    vector<int> recv_counts(nprocs);
    MPI_Allgather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displs(nprocs);
    int total_count = 0;
    for (int p = 0; p < nprocs; p++) {
        displs[p] = total_count;
        total_count += recv_counts[p];
    }

    if (total_count == 0) return;

    vector<int> all_new_nodes(total_count);
    MPI_Allgatherv(local_new_nodes.data(), local_count, MPI_INT, all_new_nodes.data(), recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    for (int node : all_new_nodes) {
        global_partitioned.insert(node);
    }
}

void print_summary(int procId, int nprocs, const vector<vector<int>> &partitions, const unordered_map<int, int> &global_degree) {
    if (procId != 0) return;

    cout << "\n=== Partitioning Summary ===\n";
    cout << "Number of partitions: " << partitions.size() << "\n";
    cout << "Total nodes in graph: " << global_degree.size() << "\n";

    int total_partitioned = 0;
    size_t min_size = SIZE_MAX;
    size_t max_size = 0;
    vector<size_t> partition_sizes;

    for (size_t i = 0; i < partitions.size(); i++) {
        unordered_set<int> unique_nodes(partitions[i].begin(), partitions[i].end());
        size_t size = unique_nodes.size();
        partition_sizes.push_back(size);
        
        total_partitioned += static_cast<int>(size);
        min_size = min(min_size, size);
        max_size = max(max_size, size);

        cout << "Partition " << i << ": " << size << " nodes";
        if (size != partitions[i].size()) {
            cout << " (original: " << partitions[i].size() << ", duplicates removed)";
        }
        cout << "\n";
    }

    cout << "Total partitioned nodes: " << total_partitioned << "\n";
    cout << "Min partition size: " << min_size << "\n";
    cout << "Max partition size: " << max_size << "\n";
    
    if (total_partitioned > 0 && !partitions.empty()) {
        double avg_size = static_cast<double>(total_partitioned) / partitions.size();
        double imbalance = static_cast<double>(max_size) / avg_size;
        
        double variance = 0.0;
        for (size_t size : partition_sizes) {
            variance += (static_cast<double>(size) - avg_size) * (static_cast<double>(size) - avg_size);
        }
        variance /= partitions.size();
        double std_dev = sqrt(variance);
        
        cout << "Average partition size: " << avg_size << "\n";
        cout << "Standard deviation: " << std_dev << "\n";
        cout << "Load imbalance ratio: " << imbalance << "\n";
        cout << "Coverage: " << (static_cast<double>(total_partitioned) / global_degree.size() * 100.0) << "%\n";
    }
    
    cout << "=======================================\n\n";
}

void printComparisonReport(const PartitioningMetrics& m1, const PartitioningMetrics& m2) {
    std::cout << "\n┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    메트릭 비교 결과                         │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ 그래프 정보:                                                │\n";
    std::cout << "│   총 노드 수:     " << std::setw(10) << m1.total_vertices << "                              │\n";
    std::cout << "│   총 간선 수:     " << std::setw(10) << m1.total_edges << "                              │\n";
    std::cout << "│   파티션 수:      " << std::setw(10) << m1.num_partitions << "                              │\n";
    std::cout << "│                                                             │\n";
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