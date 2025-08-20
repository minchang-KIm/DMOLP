#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstddef>
#include <iostream>

#include "graph_types.h"
#include "phase1/random.hpp"  // 빌드 환경에 맞게 유지

using std::unordered_map;
using std::unordered_set;
using std::vector;

// === 유틸리티: 정렬 후 중복 제거 ===
template <class T>
static inline void sort_unique(vector<T>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

// === 모듈로 기반 초기 분할 ===
// 헤더(phase1/partition.hpp)의 선언과 동일한 시그니처를 가정합니다.
void random_partition(
    int mpi_rank,
    int mpi_size,
    int num_parts,
    int theta,  // 미사용
    const std::vector<int>& seeds,  // 미사용
    const std::unordered_map<int,int>& global_degree, // 미사용
    const std::unordered_map<int, std::vector<int>>& graph, // 로컬에 로드된 adj
    Graph& local_graph,
    GhostNodes& ghost_nodes,
    bool verbose
) {
    // 0) 초기화
    local_graph.clear();

    // GhostNodes에 clear()가 없다면 명시적으로 필드 초기화
    ghost_nodes.global_to_local.clear();
    ghost_nodes.global_ids.clear();
    ghost_nodes.ghost_labels.clear();

    // 1) 로컬 정점 집합 생성(정렬/유니크)
    vector<int> locals;
    locals.reserve(graph.size());
    for (const auto& kv : graph) locals.push_back(kv.first);
    sort_unique(locals);

    const int L = (int)locals.size();
    const int ghost_base = L;

    // 2) 전역→로컬 인덱스 맵
    unordered_map<int,int> local_g2l;
    local_g2l.reserve(L * 2);
    for (int i = 0; i < L; ++i) local_g2l[locals[i]] = i;

    // 3) Graph 메타 채우기(로컬 파트)
    local_graph.num_vertices = L;                 // 행 수(=로컬 정점 수)
    local_graph.global_ids   = locals;            // 0..L-1의 전역ID
    local_graph.vertex_labels.assign(L, -1);      // 로컬 라벨만 우선 채움
    local_graph.row_ptr.assign(L + 1, 0);
    local_graph.col_indices.clear();

    // (대략적 reserve: 로컬 차수 합)
    size_t est_edges = 0;
    for (int gid : locals) {
        auto it = graph.find(gid);
        if (it != graph.end()) est_edges += it->second.size();
    }
    local_graph.col_indices.reserve(est_edges);

    // 4) 라벨 부여: global_id % num_parts
    for (int li = 0; li < L; ++li) {
        const int gid = local_graph.global_ids[li];
        local_graph.vertex_labels[li] = (num_parts > 0) ? (gid % num_parts) : 0;
    }

    // 5) CSR 생성 + 고스트 등록
    //    - col_indices에는 [0..L-1]=로컬, [L..L+G-1]=고스트 인덱스가 혼재
    for (int li = 0; li < L; ++li) {
        const int u_global = local_graph.global_ids[li];
        local_graph.row_ptr[li] = (int)local_graph.col_indices.size();

        auto it = graph.find(u_global);
        if (it != graph.end()) {
            // 이웃 정렬/중복제거 및 자기루프 제거
            vector<int> nbrs = it->second;
            sort_unique(nbrs);
            nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), u_global), nbrs.end());

            for (int v_global : nbrs) {
                auto lit = local_g2l.find(v_global);
                if (lit != local_g2l.end()) {
                    // 로컬 이웃
                    local_graph.col_indices.push_back(lit->second);
                } else {
                    // 고스트 이웃: 신규 등록 시 메타 추가
                    auto git = ghost_nodes.global_to_local.find(v_global);
                    int gidx;
                    if (git == ghost_nodes.global_to_local.end()) {
                        gidx = (int)ghost_nodes.global_ids.size();
                        ghost_nodes.global_to_local[v_global] = gidx;
                        ghost_nodes.global_ids.push_back(v_global);
                        // 고스트 라벨도 동일 규칙으로 즉시 결정 가능
                        const int glabel = (num_parts > 0) ? (v_global % num_parts) : 0;
                        ghost_nodes.ghost_labels.push_back(glabel);
                    } else {
                        gidx = git->second;
                    }
                    local_graph.col_indices.push_back(ghost_base + gidx);
                }
            }
        }
    }
    local_graph.row_ptr[L] = (int)local_graph.col_indices.size();
    local_graph.num_edges  = (int)local_graph.col_indices.size();

    std::cout << "[random_partition] Rank " << mpi_rank
                << " | locals=" << L
                << " edges=" << local_graph.num_edges
                << " ghosts=" << ghost_nodes.global_ids.size()
                << "\n";

    // 미사용 인자 경고 방지
    (void)mpi_rank; (void)mpi_size; (void)theta; (void)seeds; (void)global_degree; (void)verbose;
}
