#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <queue>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <mpi.h>
#include "types.h"
#include "phase1.h"

// --- 전역 그래프 정보 저장용 ---
static int g_global_num_vertices = 0;
static int g_global_num_edges = 0;
int get_global_num_vertices() { return g_global_num_vertices; }
int get_global_num_edges() { return g_global_num_edges; }

// --- 외부 함수 선언 (정의는 다른 파일에 있음) ---
std::vector<int> find_hub_nodes(const std::unordered_map<int, int>& global_degree);
std::vector<int> find_landmarks(const std::unordered_map<int, int>& global_degree);
std::vector<int> find_seeds(int procId, int nprocs, int numParts, const std::vector<int>& landmarks, const std::vector<int>& hub_nodes, const std::unordered_map<int, std::vector<int>>& adj);
// --- 파티션별 CSR 추출 ---
void extract_partition_csr(const Graph& full_graph, const std::vector<int>& all_labels, int mpi_rank, Graph& local_graph, std::vector<int>& local_vertices) {
    // local_vertices: mpi_rank에 해당하는 파티션 노드 인덱스
    local_vertices.clear();
    for (int u = 0; u < (int)all_labels.size(); ++u) {
        if (all_labels[u] == mpi_rank) local_vertices.push_back(u);
    }
    int n = local_vertices.size();
    local_graph.num_vertices = n;
    local_graph.row_ptr.assign(n + 1, 0);
    std::unordered_map<int, int> global2local;
    for (int i = 0; i < n; ++i) global2local[local_vertices[i]] = i;
    // 엣지 추출
    for (int i = 0; i < n; ++i) {
        int u = local_vertices[i];
        for (int idx = full_graph.row_ptr[u]; idx < full_graph.row_ptr[u+1]; ++idx) {
            int v = full_graph.col_indices[idx];
            if (global2local.count(v)) {
                local_graph.col_indices.push_back(global2local[v]);
                local_graph.row_ptr[i+1]++;
            }
        }
    }
    for (int i = 1; i <= n; ++i)
        local_graph.row_ptr[i] += local_graph.row_ptr[i-1];
    local_graph.num_edges = local_graph.col_indices.size();
}

// --- 파티션 메트릭 계산 ---
void compute_metrics(const Graph& full_graph, const std::vector<int>& all_labels, int num_partitions, Phase1Metrics& metrics) {
    metrics.total_vertices = full_graph.num_vertices;
    metrics.total_edges = full_graph.num_edges;
    metrics.partition_vertex_counts.assign(num_partitions, 0);
    metrics.partition_edge_counts.assign(num_partitions, 0);
    for (int u = 0; u < full_graph.num_vertices; ++u) {
        int p = all_labels[u];
        if (p >= 0 && p < num_partitions) {
            metrics.partition_vertex_counts[p]++;
            metrics.partition_edge_counts[p] += full_graph.row_ptr[u+1] - full_graph.row_ptr[u];
        }
    }
    // 더미: cut, balance 등은 0으로
    metrics.initial_edge_cut = 0;
    metrics.initial_vertex_balance = 0;
    metrics.initial_edge_balance = 0;
}
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <queue>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <mpi.h>
#include "types.h"
#include "phase1.h"
// 1. 파일에서 그래프 로드 (edge list, CSR 변환)
bool loadGraphFromFile(const std::string& filename, Graph& graph, std::vector<int>& vertex_labels, int num_partitions) {
    std::ifstream fin(filename);
    if (!fin.is_open()) return false;
    int max_v = 0, u, v;
    std::vector<std::pair<int, int>> edges;
    while (fin >> u >> v) {
        edges.emplace_back(u, v);
        max_v = std::max({max_v, u, v});
    }
    graph.num_vertices = max_v + 1;
    graph.num_edges = edges.size();
    graph.row_ptr.assign(graph.num_vertices + 1, 0);
    for (auto& e : edges) graph.row_ptr[e.first + 1]++;
    for (int i = 1; i <= graph.num_vertices; ++i)
        graph.row_ptr[i] += graph.row_ptr[i - 1];
    graph.col_indices.resize(edges.size());
    std::vector<int> cur(graph.num_vertices, 0);
    for (auto& e : edges) {
        int pos = graph.row_ptr[e.first] + cur[e.first];
        graph.col_indices[pos] = e.second;
        cur[e.first]++;
    }
    // vertex_labels: 균등 분할 (테스트용)
    vertex_labels.assign(graph.num_vertices, -1);
    int per_part = (graph.num_vertices + num_partitions - 1) / num_partitions;
    for (int i = 0; i < graph.num_vertices; ++i)
        vertex_labels[i] = i / per_part;
    return true;
}

// 2. MPI 기반 그래프 분배 (간단: 각 rank가 자신 파티션만 보유)
Graph distributeGraphViaMPI(const Graph& global_graph, const std::vector<int>& global_vertex_labels, std::vector<int>& local_vertex_labels, int mpi_rank, int mpi_size) {
    int num_vertices = global_graph.num_vertices;
    int num_partitions = mpi_size;
    int per_part = (num_vertices + num_partitions - 1) / num_partitions;
    int start = mpi_rank * per_part;
    int end = std::min(num_vertices, start + per_part);
    Graph local_graph;
    local_graph.num_vertices = end - start;
    local_graph.row_ptr.assign(local_graph.num_vertices + 1, 0);
    std::vector<int> mapping(num_vertices, -1);
    for (int i = start; i < end; ++i) mapping[i] = i - start;
    // local row_ptr/col_indices
    for (int u = start; u < end; ++u) {
        int local_u = mapping[u];
        for (int idx = global_graph.row_ptr[u]; idx < global_graph.row_ptr[u+1]; ++idx) {
            int v = global_graph.col_indices[idx];
            if (mapping[v] != -1) {
                local_graph.col_indices.push_back(mapping[v]);
                local_graph.row_ptr[local_u + 1]++;
            }
        }
    }
    for (int i = 1; i <= local_graph.num_vertices; ++i)
        local_graph.row_ptr[i] += local_graph.row_ptr[i-1];
    // local vertex_labels
    local_vertex_labels.resize(local_graph.num_vertices);
    for (int i = 0; i < local_graph.num_vertices; ++i)
        local_vertex_labels[i] = global_vertex_labels[start + i];
    local_graph.num_edges = local_graph.col_indices.size();
    return local_graph;
}

// 3. Phase1 메트릭 계산 (간단 예시)
Phase1Metrics calculatePhase1Metrics(const Graph& global_graph, const std::vector<int>& vertex_labels, int num_partitions, long loading_time_ms, long distribution_time_ms) {
    Phase1Metrics m = {};
    m.total_vertices = global_graph.num_vertices;
    m.total_edges = global_graph.num_edges;
    // 파티션별 노드/엣지 카운트
    m.partition_vertex_counts.assign(num_partitions, 0);
    m.partition_edge_counts.assign(num_partitions, 0);
    for (int u = 0; u < global_graph.num_vertices; ++u) {
        int p = vertex_labels[u];
        if (p >= 0 && p < num_partitions) {
            m.partition_vertex_counts[p]++;
            m.partition_edge_counts[p] += global_graph.row_ptr[u+1] - global_graph.row_ptr[u];
        }
    }
    m.loading_time_ms = loading_time_ms;
    m.distribution_time_ms = distribution_time_ms;
    // 기타 메트릭(임시)
    m.initial_edge_cut = 0;
    m.initial_vertex_balance = 0;
    m.initial_edge_balance = 0;
    return m;
}
// Phase1 메인 함수: rank가 여러 파티션을 담당할 수 있도록 part_id만 담당하는 분배로직
Phase1Metrics phase1_partition_and_distribute(int part_id, int num_partitions, int total_partitions, Graph& local_graph, std::vector<int>& vertex_labels, const std::string& filename) {
    int total_vertices = 0;
    Graph full_graph;
    loadGraphFromFile(filename, full_graph, vertex_labels, total_partitions); // vertex_labels는 임시, 아래서 재설정
    g_global_num_vertices = full_graph.num_vertices;
    g_global_num_edges = full_graph.num_edges;

    // 1. degree 벡터, adj 리스트 생성 (모든 노드에 대해 adj[u]를 반드시 생성)
    std::unordered_map<int, int> global_degree;
    std::unordered_map<int, std::vector<int>> adj;
    for (int u = 0; u < full_graph.num_vertices; ++u) {
        adj[u] = {};
        int deg = full_graph.row_ptr[u+1] - full_graph.row_ptr[u];
        global_degree[u] = deg;
        for (int idx = full_graph.row_ptr[u]; idx < full_graph.row_ptr[u+1]; ++idx) {
            adj[u].push_back(full_graph.col_indices[idx]);
        }
    }

    // 2. 허브/랜드마크/시드 탐색 (모든 파티션 기준)
    std::vector<int> hub_nodes = find_hub_nodes(global_degree);
    std::vector<int> landmarks = find_landmarks(global_degree);
    std::vector<int> seeds = find_seeds(0, total_partitions, total_partitions, landmarks, hub_nodes, adj);

    // 3. seeds 기반 vertex_labels 생성 (모든 노드 -1로 초기화, seeds는 각 파티션 id로)
    std::vector<int> all_labels(full_graph.num_vertices, -1);
    for (int p = 0; p < seeds.size(); ++p) {
        if (seeds[p] >= 0 && seeds[p] < (int)all_labels.size())
            all_labels[seeds[p]] = p;
    }
    // BFS로 각 seed에서 파티션 id 전파
    std::queue<int> q;
    for (int p = 0; p < (int)seeds.size(); ++p) {
        if (seeds[p] >= 0 && seeds[p] < (int)all_labels.size())
            q.push(seeds[p]);
    }
    while (!q.empty()) {
        int u = q.front(); q.pop();
        int p = all_labels[u];
        for (int v : adj[u]) {
            if (all_labels[v] == -1) {
                all_labels[v] = p;
                q.push(v);
            }
        }
    }

    // 4. 파티션별 CSR 추출 (part_id만 담당)
    std::vector<int> local_vertices;
    extract_partition_csr(full_graph, all_labels, part_id, local_graph, local_vertices);
    // local_graph에 맞는 vertex_labels 생성 (로컬 인덱스 기준, 인덱스 안전성 보장)
    vertex_labels.resize(local_graph.num_vertices);
    for (int i = 0; i < local_graph.num_vertices; ++i) {
        int global_idx = (i < (int)local_vertices.size()) ? local_vertices[i] : -1;
        if (global_idx >= 0 && global_idx < (int)all_labels.size())
            vertex_labels[i] = all_labels[global_idx];
        else
            vertex_labels[i] = -1;
    }

    // 5. 메트릭 계산
    Phase1Metrics metrics = {};
    compute_metrics(full_graph, all_labels, total_partitions, metrics);
    printf("[partition %d] local nodes %d\n", part_id, local_graph.num_vertices);
    return metrics;
}
// --- hub/landmark selection from partitioning/hub.cpp ---
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

using namespace std;

vector<int> find_hub_nodes(const unordered_map<int, int> &global_degree) {
    vector<pair<int, int>> sorted_degree(global_degree.begin(), global_degree.end());

    sort(sorted_degree.begin(), sorted_degree.end(), [](const auto &a, const auto &b) {
        return a.second < b.second;
    });

    int N = sorted_degree.size();
    vector<double> cum_x(N), cum_y(N);

    double total_deg = 0.0;
    for (const auto &[_, deg] : sorted_degree)
        total_deg += deg;

    double sum_deg = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_deg += sorted_degree[i].second;
        cum_x[i] = (double)(i + 1) / N;
        cum_y[i] = sum_deg / total_deg;
    }

    double x1 = cum_x[N - 1];
    double y1 = cum_y[N - 1];
    double x0 = cum_x[N - 2];
    double y0 = cum_y[N - 2];
    double slope = (y1 - y0) / (x1 - x0);

    double x_intercept = x1 - y1 / slope;

    vector<int> hub_nodes;
    for (int i = N - 1; i >= 0; --i) {
        if (cum_x[i] > x_intercept) hub_nodes.push_back(sorted_degree[i].first);
        else break;
    }

    return hub_nodes;
}

vector<int> find_landmarks(const unordered_map<int, int> &global_degree) {
    int N = global_degree.size();
    int K = max(1, (int)log10(N));

    vector<pair<int, int>> sorted_degree(global_degree.begin(), global_degree.end());
    sort(sorted_degree.begin(), sorted_degree.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    vector<int> landmarks;
    for (int i = 0; i < K && i < sorted_degree.size(); ++i)
        landmarks.push_back(sorted_degree[i].first);

    return landmarks;
}
// --- seed_greedy.cpp 기반 실제 multi-BFS 기반 시드 선택 구현 ---
const int INF = 0x3f3f3f3f;

// 허브 간 거리 계산용: landmarks, hub_nodes, adj 필요
std::vector<std::unordered_map<int, int>> compute_distances(int procId, int nprocs, const std::vector<int> &landmarks, const std::vector<int> &hub_nodes, const std::unordered_map<int, std::vector<int>> &adj) {
    int L = landmarks.size();
    int H = hub_nodes.size();
    if (L == 0 || H == 0) return {};
    int landmarks_per_proc = (L + nprocs - 1) / nprocs;
    int start_idx = procId * landmarks_per_proc;
    int end_idx = std::min(L, start_idx + landmarks_per_proc);
    int local_count = std::max(0, end_idx - start_idx);
    std::unordered_map<int, int> hub_idx;
    for (int h = 0; h < H; ++h) hub_idx[hub_nodes[h]] = h;
    std::vector<std::unordered_map<int, int>> distances(L);
    std::vector<std::vector<int>> local_results(local_count, std::vector<int>(H, INF));
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < local_count; ++i) {
        int landmark_idx = start_idx + i;
        int landmark = landmarks[landmark_idx];
        // 단일 BFS (adj 기반)
        std::queue<int> q;
        std::unordered_map<int, int> dist;
        q.push(landmark);
        dist[landmark] = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj.at(u)) {
                if (!dist.count(v)) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        for (const auto& [node, d] : dist) {
            if (hub_idx.count(node)) {
                int hub_i = hub_idx[node];
                local_results[i][hub_i] = d;
            }
        }
    }
    std::vector<int> sendbuf(L * H, INF);
    for (int i = 0; i < local_count; ++i) {
        for (int h = 0; h < H; ++h) {
            int global_idx = start_idx + i;
            sendbuf[global_idx * H + h] = local_results[i][h];
        }
    }
    std::vector<int> recvbuf(L * H, INF);
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), L * H, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h)
            if (recvbuf[l * H + h] != INF) distances[l][hub_nodes[h]] = recvbuf[l * H + h];
    }
    return distances;
}

std::vector<int> compute_hub_pairs_distances(int procId, int nprocs, const std::vector<int> &landmarks, const std::vector<int> &hub_nodes, const std::unordered_map<int, std::vector<int>> &adj) {
    auto distances = compute_distances(procId, nprocs, landmarks, hub_nodes, adj);
    int L = landmarks.size();
    int H = hub_nodes.size();
    if (L == 0 || H == 0) return {};
    std::unordered_map<int, int> hub_idx;
    for (int h = 0; h < H; ++h) hub_idx[hub_nodes[h]] = h;
    std::vector<std::vector<int>> landmarks_distances(L, std::vector<int>(H, INF));
    for (int l = 0; l < L; ++l) {
        for (const auto& [hub, d] : distances[l])
            if (hub_idx.count(hub)) landmarks_distances[l][hub_idx[hub]] = d;
    }
    int vector_size = H * (H - 1) / 2;
    std::vector<int> local(vector_size, INF);
    std::vector<int> result(vector_size, INF);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < H; ++i) {
        for (int j = i + 1; j < H; ++j) {
            int min_dist = INF;
            for (int l = 0; l < L; ++l) {
                int d1 = landmarks_distances[l][i];
                int d2 = landmarks_distances[l][j];
                if (d1 != INF && d2 != INF) min_dist = std::min(min_dist, d1 + d2);
            }
            int idx = i * H - (i + 1) * (i + 2) / 2 + j;
            local[idx] = min_dist;
        }
    }
    MPI_Allreduce(local.data(), result.data(), vector_size, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return result;
}

int hub_distance_indexed(int i, int j, int H, const std::vector<int> &distances) {
    if (i == j) return 0;
    int idx;
    if (i < j) idx = i * H - (i + 1) * (i + 2) / 2 + j;
    else idx = j * H - (j + 1) * (j + 2) / 2 + i;
    if (idx < 0 || idx >= (int)distances.size()) return INF;
    return distances[idx];
}

std::vector<int> find_seeds(const int procId, const int nprocs, const int numParts, const std::vector<int> &landmarks, const std::vector<int> &hub_nodes, const std::unordered_map<int, std::vector<int>> &adj) {
    auto distances = compute_hub_pairs_distances(procId, nprocs, landmarks, hub_nodes, adj);
    int H = hub_nodes.size();
    if (numParts > H) return {};
    if (distances.empty()) return {};
    std::vector<int> selected_hubs;
    std::vector<bool> used(H, false);
    int max_dist = -1, best_i = -1, best_j = -1;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) {
            int dist = hub_distance_indexed(i, j, H, distances);
            if (dist != INF && dist > max_dist) {
                max_dist = dist;
                best_i = i;
                best_j = j;
            }
        }
    }
    if (best_i == -1 || best_j == -1) return {};
    selected_hubs.push_back(best_i);
    selected_hubs.push_back(best_j);
    used[best_i] = true;
    used[best_j] = true;
    for (int k = 2; k < numParts; ++k) {
        int best_candidate = -1;
        long long max_total_dist = -1;
        #pragma omp parallel for schedule(dynamic)
        for (int h = 0; h < H; ++h) {
            if (used[h]) continue;
            long long total_dist = 0;
            bool valid = true;
            for (int selected : selected_hubs) {
                int dist = hub_distance_indexed(h, selected, H, distances);
                if (dist == INF) { valid = false; break; }
                total_dist += dist;
            }
            if (valid) {
                #pragma omp critical
                {
                    if (total_dist > max_total_dist) {
                        max_total_dist = total_dist;
                        best_candidate = h;
                    }
                }
            }
        }
        if (best_candidate == -1) break;
        selected_hubs.push_back(best_candidate);
        used[best_candidate] = true;
    }
    std::vector<int> result;
    for (int hub_idx : selected_hubs) result.push_back(hub_nodes[hub_idx]);
    return result;
}
#include <unordered_map>

// --- 허브/랜드마크/시드 기반 파티셔닝 함수 선언 (partitioning 코드 이식) ---
std::vector<int> find_hub_nodes(const std::unordered_map<int, int>& global_degree);
std::vector<int> find_landmarks(const std::unordered_map<int, int>& global_degree);
std::vector<int> find_seeds(const int procId, const int nprocs, const int numParts, const std::vector<int> &landmarks, const std::vector<int> &hub_nodes, const std::unordered_map<int, std::vector<int>> &adj);


#include "phase1.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <queue>
#include <mpi.h>
#include "types.h"

// 간단한 텍스트 파일 기반 그래프 로더 (edge list: u v per line, 0-based)
// 테스트용: 전체 정점을 파티션 개수만큼 균등 분할하여 각 rank가 자신의 라벨을 가진 파티션만 보유
// partitioning/main.cpp 스타일의 명령어 파싱 및 파티셔닝 로직을 types.h 기반으로 이식
// theta는 phase1에서 사용, numParts는 num_partitions로 통일

// 그래프 로더: edge list 파일을 CSR로 변환
static void load_graph_from_file(const std::string& filename, Graph& graph, int& total_vertices) {
    std::ifstream fin(filename);
    int max_v = 0;
    int u, v;
    std::vector<std::pair<int, int>> edges;
    while (fin >> u >> v) {
        edges.emplace_back(u, v);
        max_v = std::max({max_v, u, v});
    }
    total_vertices = max_v + 1;
    graph.num_vertices = total_vertices;
    graph.num_edges = edges.size();
    graph.row_ptr.assign(graph.num_vertices + 1, 0);
    for (auto& e : edges) {
        graph.row_ptr[e.first + 1]++;
    }
    for (int i = 1; i <= graph.num_vertices; ++i)
        graph.row_ptr[i] += graph.row_ptr[i - 1];
    graph.col_indices.resize(edges.size());
    std::vector<int> cur(graph.num_vertices, 0);
    for (auto& e : edges) {
        int pos = graph.row_ptr[e.first] + cur[e.first];
        graph.col_indices[pos] = e.second;
        cur[e.first]++;
    }
}
