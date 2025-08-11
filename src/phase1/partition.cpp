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

// ======== Ghost 캐시 구조(이웃 + 현재 라벨) ========
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

// 라운드 종료 시에만 전역 파티션 크기 스냅샷 갱신(로컬 소유 노드 기준 합산)
static void recompute_partition_sizes(
    int numParts,
    const std::unordered_map<int, std::vector<int>>& local_adj,
    const std::unordered_map<int, int>& node_label,
    std::vector<int>& part_size_global,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    if ((int)part_size_global.size() != numParts) {
        part_size_global.assign(numParts, 0);
    }
    std::vector<int> local_count(numParts, 0);
    for (const auto& kv : local_adj) {
        int u = kv.first;
        auto it = node_label.find(u);
        if (it != node_label.end() && it->second >= 0) {
            int p = it->second;
            if (0 <= p && p < numParts) ++local_count[p];
        }
    }
    MPI_Allreduce(local_count.data(), part_size_global.data(),
                  numParts, MPI_INT, MPI_SUM, comm);
}

// 고스트 정보(이웃 + 현재 라벨) 가져오기
static void fetch_ghost_nodes(
    const unordered_set<int> &to_request_ghosts,
    const unordered_map<int, vector<int>> &local_adj,
    const unordered_map<int, int> &node_label,   // 현재 소유 랭크의 라벨 테이블
    int procId,
    int nprocs,
    unordered_map<int, GhostEntry> &ghost_nodes  // [출력] ghost_id → {nbrs, label}
) {
    auto owner_of = [&](int gid){ return gid % nprocs; };

    // 1) 각 소유 랭크별 요청 목록
    vector<vector<int>> requests_per_rank(nprocs);
    for (int ghost : to_request_ghosts) {
        int owner = owner_of(ghost);
        requests_per_rank[owner].push_back(ghost);
    }

    // 2) 요청 카운트 교환
    vector<int> send_counts(nprocs, 0), recv_counts(nprocs, 0);
    for (int r = 0; r < nprocs; ++r) send_counts[r] = (int)requests_per_rank[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 3) 요청 버퍼 플래튼
    vector<int> sdispls(nprocs, 0), rdispls(nprocs, 0);
    int s_total = 0, r_total = 0;
    for (int r = 0; r < nprocs; ++r) { sdispls[r] = s_total; s_total += send_counts[r]; }
    for (int r = 0; r < nprocs; ++r) { rdispls[r] = r_total; r_total += recv_counts[r]; }

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
            if (degree && nbrs_ptr) for (int v : *nbrs_ptr) out.push_back(v);
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

// ======== 메인 알고리즘 ========
// - 로컬에서 선경합하지 않고, 모든 경합 요청을 소유자에게 모아 일괄 판정
// - partition_verts에는 "내가 담당하는 파티션(p % nprocs == procId)만" 저장
// - 라운드 종료 시 전역 파티션 크기 스냅샷 갱신
// - 다음 라운드 프론티어는 "이번 라운드에 새로 라벨 확정된 (로컬+고스트)"로 재등록
// - 최종 CSR: "내 파티션의 노드들"을 행으로 생성(이웃은 로컬/고스트 모두 사용)
void partition_expansion(
    int procId,
    int nprocs,
    int numParts,
    const std::vector<int> &seeds,
    const std::unordered_map<int, int> &global_degree,
    const std::unordered_map<int, std::vector<int>> &local_adj,
    Graph &local_partition_graphs,
    GhostNodes &local_partition_ghosts
) {
    const bool verbose = true;
    std::cout << ">>> ENTER partition_expansion v2 (rank=" << procId << ")\n";

    auto owner_of       = [&](int gid){ return gid % nprocs; };
    auto is_my_partition= [&](int p)  { return (p % nprocs) == procId; };

    // 상태
    std::vector<std::vector<int>> partition_verts(numParts);    // 내 파티션만 저장
    std::vector<int> part_size_global(numParts, 0);             // 라운드 말 스냅샷(전역)
    std::vector<std::queue<int>> frontiers(numParts);

    std::unordered_map<int,int> node_label;                     // 로컬 소유 노드 라벨
    std::unordered_map<int, GhostEntry> ghost_nodes;            // 고스트(이웃 + 라벨)

    std::vector<int> my_parts;
    for (auto &kv : local_adj) node_label[kv.first] = -1;

    // 시드 배치(+ 프론티어 준비)
    std::unordered_set<int> initial_ghost_seed;
    for (int p=0; p<numParts; ++p) {
        int seed = seeds[p];
        if (is_my_partition(p)) {
            my_parts.push_back(p);
            frontiers[p].push(seed);
            if (local_adj.count(seed)) {
                node_label[seed] = p;
                if (is_my_partition(p)) partition_verts[p].push_back(seed); // 내 파티션만 저장
                if (verbose) std::cout << "[R" << procId << "] seed local P" << p << " <- " << seed << "\n";
            } else {
                initial_ghost_seed.insert(seed);
                if (verbose) std::cout << "[R" << procId << "] seed ghost P" << p << " <- " << seed << "\n";
            }
        }
    }

    // 초기 고스트 페치 + 스냅샷
    fetch_ghost_nodes(initial_ghost_seed, local_adj, node_label, procId, nprocs, ghost_nodes);
    recompute_partition_sizes(numParts, local_adj, node_label, part_size_global);

    // 종료 조건(전역)
    int total_num = (int)global_degree.size();
    auto count_local_labeled = [&](){
        int c=0; for (auto &kv: node_label) if (kv.second>=0) ++c; return c;
    };
    int total_labeled=0, local_labeled=count_local_labeled();
    MPI_Allreduce(&local_labeled, &total_labeled, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int round = 0;
    while (total_labeled < total_num) {
        if (verbose) std::cout << "[R" << procId << "] ===== Round " << round << " =====\n";

        // 이번 라운드에 새로 라벨 확정된 노드(다음 프론티어 재등록용)
        std::vector<std::vector<int>> newly_labeled_local(numParts), newly_labeled_ghost(numParts);

        // 프론티어 비었으면 로컬 미라벨 균형 배분(옵션)
        bool all_empty = true;
        for (int p: my_parts) if (!frontiers[p].empty()) { all_empty=false; break; }
        if (all_empty) {
            for (auto &kv : local_adj) {
                int v = kv.first;
                auto it = node_label.find(v);
                if (it==node_label.end() || it->second<0) {
                    int best = my_parts[0];
                    for (int p: my_parts) {
                        if (part_size_global[p] < part_size_global[best] ||
                           (part_size_global[p]==part_size_global[best] && p<best)) best=p;
                    }
                    node_label[v]=best;
                    if (is_my_partition(best)) {
                        partition_verts[best].push_back(v);
                        frontiers[best].push(v);
                    }
                }
            }
        }

        // 프론티어 크기 로그
        size_t frontier_total = 0;
        for (int p : my_parts) frontier_total += frontiers[p].size();
        if (verbose) std::cout << "[R" << procId << "] round " << round << " frontier_total=" << frontier_total << "\n";

        // 프론티어 확장 → 다음 후보 수집 + 고스트 프리페치
        std::unordered_set<int> to_request_ghosts;
        std::unordered_set<int> visited; // 라운드 내 v 중복 방지
        std::vector<std::queue<int>> next_frontiers(numParts);

        for (int p : my_parts) {
            std::queue<int> q = frontiers[p];
            while (!q.empty()) {
                int u = q.front(); q.pop();
                const std::vector<int>* nbrs = nullptr;
                if (local_adj.count(u)) nbrs=&local_adj.at(u);
                else if (ghost_nodes.count(u)) nbrs=&ghost_nodes[u].nbrs;
                if (!nbrs) continue;

                for (int v : *nbrs) {
                    // 고스트 프리페치(캐시 없음 또는 라벨 미정)
                    if (!local_adj.count(v)) {
                        auto git = ghost_nodes.find(v);
                        if (git==ghost_nodes.end() || git->second.label==-1) to_request_ghosts.insert(v);
                    }
                    // 미라벨이면 후보로
                    if (is_unlabeled(node_label, ghost_nodes, local_adj, v) && visited.insert(v).second) {
                        next_frontiers[p].push(v);
                    }
                }
            }
        }
        if (verbose) std::cout << "[R" << procId << "] round " << round << " to_request_ghosts=" << to_request_ghosts.size() << "\n";

        fetch_ghost_nodes(to_request_ghosts, local_adj, node_label, procId, nprocs, ghost_nodes);

        // 경합 요청 생성: 로컬 후보도 포함하여 소유자에게 모두 보냄
        // 형식: [v, |plist|, plist...]
        std::vector<std::vector<int>> req_per_owner(nprocs);
        for (int p=0; p<numParts; ++p) {
            while (!next_frontiers[p].empty()) {
                int v = next_frontiers[p].front(); next_frontiers[p].pop();
                if (!is_unlabeled(node_label, ghost_nodes, local_adj, v)) continue;
                int ow = owner_of(v);
                req_per_owner[ow].push_back(v);
                req_per_owner[ow].push_back(1);
                req_per_owner[ow].push_back(p);
            }
        }

        // Alltoallv: 경합 요청 송수신
        std::vector<int> send_counts(nprocs,0), recv_counts(nprocs,0);
        for (int r=0;r<nprocs;++r) send_counts[r] = (int)req_per_owner[r].size();
        MPI_Alltoall(send_counts.data(),1,MPI_INT, recv_counts.data(),1,MPI_INT, MPI_COMM_WORLD);

        int recv_req_total = 0; for (int r=0;r<nprocs;++r) recv_req_total += recv_counts[r];
        if (verbose) std::cout << "[R" << procId << "] round " << round << " recv_counts(req)_sum=" << recv_req_total << "\n";

        std::vector<int> sdis(nprocs,0), rdis(nprocs,0);
        int st=0, rt=0;
        for (int r=0;r<nprocs;++r){ sdis[r]=st; st+=send_counts[r]; }
        for (int r=0;r<nprocs;++r){ rdis[r]=rt; rt+=recv_counts[r]; }
        std::vector<int> sendbuf(st), recvbuf(rt);
        for (int r=0, idx=0; r<nprocs; ++r) for (int x: req_per_owner[r]) sendbuf[idx++]=x;

        MPI_Alltoallv(sendbuf.data(), send_counts.data(), sdis.data(), MPI_INT,
                      recvbuf.data(), recv_counts.data(), rdis.data(), MPI_INT, MPI_COMM_WORLD);

        // 소유자에서 일괄 판정
        std::unordered_map<int, std::vector<int>> plist_by_v;      // v -> 후보 파티션들
        std::unordered_map<int, std::vector<int>> requesters_by_v; // v -> 요청자 랭크들
        for (int r=0; r<nprocs; ++r) {
            int start = rdis[r], end = rdis[r]+recv_counts[r];
            for (int i=start; i<end; ) {
                int v   = recvbuf[i++];
                int len = recvbuf[i++];
                if (owner_of(v) != procId) { i += len; continue; }
                auto &plist = plist_by_v[v];
                auto &rq    = requesters_by_v[v];
                for (int k=0; k<len; ++k) {
                    int p = recvbuf[i++];
                    plist.push_back(p);
                    rq.push_back(r);
                }
            }
        }

        std::vector<std::vector<int>> upd_per_rank(nprocs); // 결과 통지용 [v,winner] 쌍
        int owned_claims = 0;
        for (auto &kv : plist_by_v) {
            int v = kv.first;
            auto &plist = kv.second;
            if (!local_adj.count(v)) continue; // 소유자는 반드시 로컬
            int winner = plist[0];
            for (int p : plist) {
                if (part_size_global[p] < part_size_global[winner] ||
                   (part_size_global[p]==part_size_global[winner] && p < winner)) winner = p;
            }
            // 소유자 측 반영(내 파티션일 때만 저장)
            node_label[v] = winner;
            if (is_my_partition(winner)) {
                partition_verts[winner].push_back(v);
                newly_labeled_local[winner].push_back(v);
            }
            // 모든 요청자 + 자신에게 결과 통지
            auto &rq = requesters_by_v[v];
            for (int rr : rq) { upd_per_rank[rr].push_back(v); upd_per_rank[rr].push_back(winner); }
            upd_per_rank[procId].push_back(v); upd_per_rank[procId].push_back(winner);
            ++owned_claims;
        }

        // 결과 통지 송수신
        std::vector<int> u_send_counts(nprocs,0), u_recv_counts(nprocs,0);
        for (int r=0;r<nprocs;++r) u_send_counts[r]=(int)upd_per_rank[r].size();
        MPI_Alltoall(u_send_counts.data(),1,MPI_INT, u_recv_counts.data(),1,MPI_INT, MPI_COMM_WORLD);

        std::vector<int> u_sdis(nprocs,0), u_rdis(nprocs,0);
        int ust=0, urt=0;
        for (int r=0;r<nprocs;++r){ u_sdis[r]=ust; ust+=u_send_counts[r]; }
        for (int r=0;r<nprocs;++r){ u_rdis[r]=urt; urt+=u_recv_counts[r]; }
        std::vector<int> u_sendbuf(ust), u_recvbuf(urt);
        for (int r=0, idx=0; r<nprocs; ++r) for (int x: upd_per_rank[r]) u_sendbuf[idx++]=x;

        MPI_Alltoallv(u_sendbuf.data(), u_send_counts.data(), u_sdis.data(), MPI_INT,
                      u_recvbuf.data(), u_recv_counts.data(), u_rdis.data(), MPI_INT, MPI_COMM_WORLD);

        int upd_recv_pairs = urt / 2;

        // 요청자 측 반영(고스트 포함, 단 내 파티션만 저장)
        for (int i=0; i<urt; ) {
            int v      = u_recvbuf[i++];
            int winner = u_recvbuf[i++];
            if (!local_adj.count(v)) {
                auto &ge = ghost_nodes[v]; // default 생성
                ge.label = winner;
                if (is_my_partition(winner)) {
                    partition_verts[winner].push_back(v);
                    newly_labeled_ghost[winner].push_back(v);
                }
            }
        }

        if (verbose) {
            std::cout << "[R" << procId << "] round " << round
                      << " owned_claims=" << owned_claims
                      << " upd_recv=" << upd_recv_pairs << "\n";
        }

        // 종료 검사
        local_labeled = count_local_labeled();
        MPI_Allreduce(&local_labeled, &total_labeled, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // 스냅샷 갱신 + 다음 라운드 프론티어 구성(이번 라운드 확정분)
        recompute_partition_sizes(numParts, local_adj, node_label, part_size_global);

        for (int p=0; p<numParts; ++p) {
            std::queue<int> q;
            for (int u : newly_labeled_local[p])  q.push(u);
            for (int u : newly_labeled_ghost[p])  q.push(u);
            frontiers[p].swap(q);
        }
        size_t next_frontier_total = 0;
        for (int p=0; p<numParts; ++p) next_frontier_total += frontiers[p].size();
        if (verbose) std::cout << "[R" << procId << "] round " << round
                               << " next_frontier_total=" << next_frontier_total << "\n";

        ++round;
    }

    // 최종 partition_verts 요약(내 파티션만 출력)
    std::cout << "[Rank " << procId << "] partition_verts summary:\n";
    for (int p = 0; p < (int)partition_verts.size(); ++p) {
        if (!is_my_partition(p)) continue;
        std::cout << "  Partition " << p << " (count=" << partition_verts[p].size() << "): ";
        std::cout << "\n";
    }

    // ===== 글로벌 CSR 구성(행은 '내 파티션'에 속한 노드만 포함; 이웃은 로컬/고스트 모두 사용) =====
    local_partition_graphs.num_vertices = 0;
    local_partition_graphs.num_edges = 0;
    local_partition_graphs.global_ids.clear();
    local_partition_graphs.vertex_labels.clear();
    local_partition_graphs.col_indices.clear();
    local_partition_graphs.row_ptr.clear();
    local_partition_graphs.row_ptr.push_back(0);

    auto ensure_ghost = [&](int gid) {
        if (local_partition_ghosts.global_to_local.count(gid)) return;
        int idx = (int)local_partition_ghosts.global_ids.size();
        local_partition_ghosts.global_ids.push_back(gid);
        int glabel = -1;
        if (auto itg = ghost_nodes.find(gid); itg != ghost_nodes.end()) glabel = itg->second.label;
        local_partition_ghosts.ghost_labels.push_back(glabel);
        local_partition_ghosts.global_to_local.emplace(gid, idx);
    };

    for (int p = 0; p < numParts; ++p) {
        if (!is_my_partition(p)) continue; // 내 파티션만 행 생성
        const auto &verts_in_p = partition_verts[p];
        for (int u : verts_in_p) {
            // 라벨
            int lab = -1;
            if (auto it = node_label.find(u); it != node_label.end()) lab = it->second;
            else if (auto git = ghost_nodes.find(u); git != ghost_nodes.end()) lab = git->second.label;

            // 행 등록
            local_partition_graphs.global_ids.push_back(u);
            local_partition_graphs.vertex_labels.push_back(lab);

            // 이웃: 로컬이면 local_adj, 고스트이면 ghost_nodes[u].nbrs
            const std::vector<int>* nbrs = nullptr;
            if (auto ia = local_adj.find(u); ia != local_adj.end())      nbrs = &ia->second;
            else if (auto ig = ghost_nodes.find(u); ig != ghost_nodes.end()) nbrs = &ig->second.nbrs;

            if (nbrs) {
                for (int v : *nbrs) {
                    local_partition_graphs.col_indices.push_back(v);
                    if (!local_adj.count(v)) ensure_ghost(v);
                }
            }

            local_partition_graphs.row_ptr.push_back(
                (int)local_partition_graphs.col_indices.size()
            );
        }
    }

    local_partition_graphs.num_vertices = (int)local_partition_graphs.global_ids.size();
    local_partition_graphs.num_edges    = (int)local_partition_graphs.col_indices.size();

    if (verbose) {
        std::cout << "=== [Rank " << procId << "] CSR Summary ===\n";
        std::cout << "Vertices: " << local_partition_graphs.num_vertices << "\n";
        std::cout << "Edges:    " << local_partition_graphs.num_edges << "\n";
        std::cout << "Ghosts:   " << local_partition_ghosts.global_ids.size() << "\n";
    }
}
