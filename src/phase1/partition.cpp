#include <mpi.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <iostream>

#include "graph_types.h"
#include "utils.hpp"
#include "phase1/partition.hpp"

using namespace std;

// ======== Ghost 캐시 구조 확장: 이웃 + 현재 라벨 ========
struct GhostEntry {
    std::vector<int> nbrs; // 이웃 목록
    int label = -1;        // 소유 랭크가 알고 있는 현재 라벨 (-1: 미라벨)
};

// ======== 유틸 ========
static inline bool is_unlabeled_local(const unordered_map<int,int>& node_label, int v) {
    auto it = node_label.find(v);
    return (it == node_label.end() || it->second == -1);
}

static inline bool is_unlabeled(const unordered_map<int,int>& node_label,
                                const unordered_map<int, GhostEntry>& ghost_nodes,
                                const unordered_map<int, vector<int>>& local_adj,
                                int v) {
    if (local_adj.count(v)) return is_unlabeled_local(node_label, v);
    auto it = ghost_nodes.find(v);
    if (it == ghost_nodes.end()) return true; // 아직 정보 없음 → 미라벨 취급
    return (it->second.label == -1);
}

static void fetch_ghost_nodes(
    const unordered_set<int> &to_request_ghosts,
    const unordered_map<int, vector<int>> &local_adj,
    const unordered_map<int, int> &node_label,   // 현재 소유 랭크의 라벨 테이블
    int procId,
    int nprocs,
    unordered_map<int, GhostEntry> &ghost_nodes  // [출력] ghost_id → {nbrs, label}
) {
    // 1) 각 소유 랭크별 요청 목록 구성
    vector<vector<int>> requests_per_rank(nprocs);
    for (int ghost : to_request_ghosts) {
        int owner = ghost % nprocs; // 예시: owner = gid % nprocs
        requests_per_rank[owner].push_back(ghost);
    }

    // 2) 요청 카운트 교환
    vector<int> send_counts(nprocs, 0), recv_counts(nprocs, 0);
    for (int r = 0; r < nprocs; ++r) send_counts[r] = (int)requests_per_rank[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 3) 요청 버퍼 플래튼
    vector<int> sdispls(nprocs, 0), rdispls(nprocs, 0);
    int s_total = 0, r_total = 0;
    for (int r = 0; r < nprocs; ++r) {
        sdispls[r] = s_total; s_total += send_counts[r];
        rdispls[r] = r_total; r_total += recv_counts[r];
    }
    vector<int> sendbuf(s_total), recvbuf(r_total);
    for (int r = 0, idx = 0; r < nprocs; ++r)
        for (int g : requests_per_rank[r]) sendbuf[idx++] = g;

    // 4) 요청 전송
    MPI_Alltoallv(sendbuf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                  recvbuf.data(), recv_counts.data(), rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // 5) 소유 랭크에서 응답(이웃 + 현재 라벨) 직렬화
    vector<vector<int>> send_reply_per_rank(nprocs);
    vector<int> send_reply_counts(nprocs, 0);

    for (int r = 0; r < nprocs; ++r) {
        auto &out = send_reply_per_rank[r];
        for (int j = 0; j < recv_counts[r]; ++j) {
            int nid = recvbuf[rdispls[r] + j];

            // 기본값
            int label = -1;
            int degree = 0;
            const vector<int>* nbrs_ptr = nullptr;

            auto adj_it = local_adj.find(nid);
            if (adj_it != local_adj.end()) {
                nbrs_ptr = &adj_it->second;
                degree = (int)nbrs_ptr->size();
                auto lab_it = node_label.find(nid);
                if (lab_it != node_label.end()) label = lab_it->second;
            }

            out.push_back(nid);
            out.push_back(degree);
            out.push_back(label);
            if (degree && nbrs_ptr) {
                for (int v : *nbrs_ptr) out.push_back(v);
            }
        }
        send_reply_counts[r] = (int)out.size();
    }

    // 6) 응답 크기 교환
    vector<int> recv_reply_counts(nprocs, 0);
    MPI_Alltoall(send_reply_counts.data(), 1, MPI_INT,
                 recv_reply_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 7) 응답 버퍼 플래튼 + 전송
    vector<int> reply_displs(nprocs, 0), recv_reply_displs(nprocs, 0);
    int reply_total = 0, recv_reply_total = 0;
    for (int r = 0; r < nprocs; ++r) {
        reply_displs[r] = reply_total; reply_total += send_reply_counts[r];
        recv_reply_displs[r] = recv_reply_total; recv_reply_total += recv_reply_counts[r];
    }
    vector<int> send_reply(reply_total), recv_reply(recv_reply_total);

    for (int r = 0, idx = 0; r < nprocs; ++r)
        for (int v : send_reply_per_rank[r]) send_reply[idx++] = v;

    MPI_Alltoallv(send_reply.data(), send_reply_counts.data(), reply_displs.data(), MPI_INT,
                  recv_reply.data(), recv_reply_counts.data(), recv_reply_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // 8) 역직렬화하여 ghost_nodes 캐시 갱신
    for (int i = 0; i < recv_reply_total; ) {
        int node_id = recv_reply[i++];
        int degree  = recv_reply[i++];
        int label   = recv_reply[i++];

        vector<int> nbrs;
        nbrs.reserve(degree);
        for (int d = 0; d < degree; ++d) nbrs.push_back(recv_reply[i++]);

        ghost_nodes[node_id] = GhostEntry{ std::move(nbrs), label };
    }
}

void partition_expansion(
    int procId, 
    int nprocs, 
    int numParts, 
    const vector<int> &seeds, 
    const unordered_map<int, int> &global_degree,
    const unordered_map<int, vector<int>> &local_adj, 
    unordered_map<int, Graph> &local_partition_graphs, 
    unordered_map<int, GhostNodes> &local_partition_ghosts
) {
    int total_num = global_degree.size();
    vector<vector<int>> partition_verts(numParts);
    vector<queue<int>> frontiers(numParts);

    unordered_map<int, int> node_label;
    unordered_map<int, GhostEntry> ghost_nodes;

    vector<int> my_partitions;

    for (const auto &kv : local_adj) node_label[kv.first] = -1;

    unordered_set<int> initial_ghost_seed;
    for (int p = 0; p < numParts; ++p) {
        int seed = seeds[p];
        if (p % nprocs == procId) {
            if (local_adj.count(seed)) {
                node_label[seed] = p;
                partition_verts[p].push_back(seed);
            } else {
                initial_ghost_seed.insert(seed);
            }
            frontiers[p].push(seed);
            my_partitions.push_back(p);
        }
    }
    fetch_ghost_nodes(initial_ghost_seed, local_adj, node_label, procId, nprocs, ghost_nodes);

    int local_labeled = 0;
    for (const auto &[gid, lbl] : node_label) if (lbl != -1) ++local_labeled;
    int total_labeled = 0;
    MPI_Allreduce(&local_labeled, &total_labeled, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int round = 0;
    while (total_labeled < total_num) {

        bool all_empty = true;

        cout << all_empty << endl;
        //프론티어 없는데 남는게 있는지 검사
        if (all_empty) {
            for (const auto &[v, _] : local_adj) {
                if (node_label.count(v) == 0 || node_label[v] == -1) {
                    int best_p = my_partitions[0];
                    for (int p : my_partitions) {
                        if (partition_verts[p].size() < partition_verts[best_p].size()) best_p = p;
                    }
                    node_label[v] = best_p;
                    partition_verts[best_p].push_back(v);
                    frontiers[best_p].push(v);
                }
            }
        }

        unordered_set<int> to_request_ghosts;
        for (int p : my_partitions) {
            queue<int> q = frontiers[p];
            while (!q.empty()) {
                int u = q.front(); q.pop();

                const vector<int>* nbrs_ptr = nullptr;
                if (local_adj.count(u)) nbrs_ptr = &local_adj.at(u);
                else if (ghost_nodes.count(u)) nbrs_ptr = &ghost_nodes[u].nbrs;
                if (!nbrs_ptr) continue;

                for (int v : *nbrs_ptr) {
                    if ((is_unlabeled(node_label, ghost_nodes, local_adj, v)) &&
                        !local_adj.count(v) && !ghost_nodes.count(v)) {
                        to_request_ghosts.insert(v);
                    }
                }
            }
        }
        fetch_ghost_nodes(to_request_ghosts, local_adj, node_label, procId, nprocs, ghost_nodes);

        vector<queue<int>> next_frontiers(numParts);
        unordered_set<int> visited_this_round;
        for (int p : my_partitions) {
            queue<int> q = frontiers[p];
            while (!q.empty()) {
                int u = q.front(); q.pop();

                const vector<int>* nbrs_ptr = nullptr;
                if (local_adj.count(u)) nbrs_ptr = &local_adj.at(u);
                else if (ghost_nodes.count(u)) nbrs_ptr = &ghost_nodes[u].nbrs;
                if (!nbrs_ptr) continue;

                for (int v : *nbrs_ptr) {
                    if (is_unlabeled(node_label, ghost_nodes, local_adj, v) &&
                        visited_this_round.insert(v).second) {
                        next_frontiers[p].push(v);
                    }
                }
            }
        }


        for (int p = 0; p < numParts; ++p) {
            frontiers[p] = queue<int>();
        }

        unordered_map<int, vector<int>> node_partition_candidates;
        for (int p = 0; p < numParts; ++p) {
            while (!next_frontiers[p].empty()) {
                int v = next_frontiers[p].front(); next_frontiers[p].pop();
                frontiers[p].push(v);
                node_partition_candidates[v].push_back(p);
            }
        }

        unordered_map<int, vector<pair<int, vector<int>>>> ghost_requests_per_rank;
        for (const auto &[v, plist] : node_partition_candidates) {
            if (!is_unlabeled(node_label, ghost_nodes, local_adj, v)) continue;

            if (local_adj.count(v)) {
                int winner = plist[0];
                for (int p : plist) {
                    if (partition_verts[p].size() < partition_verts[winner].size()) winner = p;
                    else if (partition_verts[p].size() == partition_verts[winner].size() && p < winner) winner = p;
                }
                node_label[v] = winner;
                partition_verts[winner].push_back(v);
            } else {
                int owner = v % nprocs;
                ghost_requests_per_rank[owner].emplace_back(v, plist);
            }
        }

        vector<int> send_counts(nprocs), recv_counts(nprocs);
        vector<vector<int>> sendbufs(nprocs);

        for (int r = 0; r < nprocs; ++r) {
            for (const auto &[v, plist] : ghost_requests_per_rank[r]) {
                sendbufs[r].push_back(v);
                sendbufs[r].push_back(plist.size());
                for (int p : plist) sendbufs[r].push_back(p);
            }
            send_counts[r] = sendbufs[r].size();
        }

        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
                     
        MPI_Barrier(MPI_COMM_WORLD);

        vector<int> sdispls(nprocs), rdispls(nprocs);
        int s_total = 0, r_total = 0;
        for (int r = 0; r < nprocs; ++r) {
            sdispls[r] = s_total; s_total += send_counts[r];
            rdispls[r] = r_total; r_total += recv_counts[r];
        }
        vector<int> sendbuf(s_total), recvbuf(r_total);
        for (int r = 0, idx = 0; r < nprocs; ++r) {
            for (int v : sendbufs[r]) sendbuf[idx++] = v;
        }

        MPI_Alltoallv(sendbuf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                      recvbuf.data(), recv_counts.data(), rdispls.data(), MPI_INT,
                      MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        int i = 0;
        while (i < recvbuf.size()) {
            int v = recvbuf[i++];
            int len = recvbuf[i++];
            vector<int> plist;
            for (int j = 0; j < len; ++j) plist.push_back(recvbuf[i++]);

            if (!local_adj.count(v)) continue;

            int winner = plist[0];
            for (int p : plist) {
                if (partition_verts[p].size() < partition_verts[winner].size()) winner = p;
                else if (partition_verts[p].size() == partition_verts[winner].size() && p < winner) winner = p;
            }
            node_label[v] = winner;
            partition_verts[winner].push_back(v);
        }

        vector<vector<int>> label_updates_per_rank(nprocs);
        vector<int> upd_send_counts(nprocs, 0);

        for (int r = 0; r < nprocs; ++r) {
            int start = rdispls[r];
            int end   = rdispls[r] + recv_counts[r];
            auto &ub  = label_updates_per_rank[r];
            for (int i = start; i < end; ) {
                int v   = recvbuf[i++];
                int len = recvbuf[i++];
                i += len;
                
                auto it = node_label.find(v);
                int winner = (it != node_label.end() ? it->second : -1);

                ub.push_back(v);
                ub.push_back(winner);
            }
            upd_send_counts[r] = (int)label_updates_per_rank[r].size();
        }

        vector<int> upd_recv_counts(nprocs, 0);
        MPI_Alltoall(upd_send_counts.data(), 1, MPI_INT,
                     upd_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        vector<int> upd_sdispls(nprocs, 0), upd_rdispls(nprocs, 0);
        int upd_stotal = 0, upd_rtotal = 0;
        for (int r = 0; r < nprocs; ++r) { upd_sdispls[r] = upd_stotal; upd_stotal += upd_send_counts[r]; }
        for (int r = 0; r < nprocs; ++r) { upd_rdispls[r] = upd_rtotal; upd_rtotal += upd_recv_counts[r]; }

        vector<int> upd_sendbuf(upd_stotal), upd_recvbuf(upd_rtotal);
        for (int r = 0, idx = 0; r < nprocs; ++r)
            for (int x : label_updates_per_rank[r]) upd_sendbuf[idx++] = x;

        MPI_Alltoallv(upd_sendbuf.data(), upd_send_counts.data(), upd_sdispls.data(), MPI_INT,
                      upd_recvbuf.data(), upd_recv_counts.data(), upd_rdispls.data(), MPI_INT,
                      MPI_COMM_WORLD);

        // === (H) [요청자] 수신 → 고스트 캐시 라벨 갱신 ===
        for (int i = 0; i < upd_rtotal; ) {
            int v      = upd_recvbuf[i++];
            int winner = upd_recvbuf[i++];
            if (!local_adj.count(v)) {
                auto &ge = ghost_nodes[v]; // 없으면 default 생성
                ge.label = winner;
                // ge.nbrs는 필요 시 fetch 때 이미 채워짐
            }
        }


        local_labeled = 0;
        for (const auto &[gid, lbl] : node_label)
            if (lbl != -1) ++local_labeled;
        MPI_Allreduce(&local_labeled, &total_labeled, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        ++round;
    }
        // =========================
    // 내부 변환 함수(lambda) 추가
    // =========================
    auto build_partition_graphs_and_ghosts = [&]() {
        // owner 판별 람다
        auto owner_of = [&](int gid) { return gid % nprocs; };

        // 출력 컨테이너 초기화
        for (int p = 0; p < numParts; ++p) {
            local_partition_graphs[p].clear();
            local_partition_ghosts[p].clear();
        }

        // 파티션별 로컬 정점 수집
        unordered_map<int, vector<int>> part_vertices;
        part_vertices.reserve(numParts);
        for (const auto &kv : local_adj) {
            int u = kv.first;
            auto it = node_label.find(u);
            if (it == node_label.end() || it->second < 0) continue;
            int p = it->second;
            part_vertices[p].push_back(u);
        }

        // 각 파티션에 대해 Graph/GhostNodes 구성
        for (int p = 0; p < numParts; ++p) {
            auto itv = part_vertices.find(p);
            if (itv == part_vertices.end() || itv->second.empty()) continue;

            vector<int> locals = itv->second;
            sort(locals.begin(), locals.end());

            Graph &G = local_partition_graphs[p];
            GhostNodes &GN = local_partition_ghosts[p];

            // Graph 기본 필드
            G.num_vertices = (int)locals.size();
            G.global_ids   = locals;
            G.vertex_labels.assign(G.num_vertices, p);
            G.row_ptr.assign(G.num_vertices + 1, 0);
            G.col_indices.clear();
            G.num_edges = 0;

            // 고스트 등록 헬퍼
            auto ensure_ghost = [&](int gid) {
                if (GN.global_to_local.find(gid) != GN.global_to_local.end()) return;
                int idx = (int)GN.global_ids.size();
                GN.global_ids.push_back(gid);
                int glabel = -1;
                auto git = ghost_nodes.find(gid);
                if (git != ghost_nodes.end()) glabel = git->second.label;
                GN.ghost_labels.push_back(glabel);
                GN.global_to_local.emplace(gid, idx);
            };

            // CSR 채우기 (col_indices는 글로벌 ID 저장)
            for (int i = 0; i < G.num_vertices; ++i) {
                int u = G.global_ids[i];
                const auto &nbrs = local_adj.at(u);
                for (int v : nbrs) {
                    G.col_indices.push_back(v);
                    ++G.num_edges;
                    if (owner_of(v) != procId) ensure_ghost(v);
                }
                G.row_ptr[i + 1] = (int)G.col_indices.size();
            }
        }
    };

    // 내부 함수 호출
    build_partition_graphs_and_ghosts();
}
