#include <mpi.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "graph_types.h"
#include "utils.hpp"
#include "phase1/partition.hpp"

using namespace std;

struct GhostEntry {
    vector<int> nbrs;
    int label = -1;
};

static bool is_unlabeled_local(const unordered_map<int,int>& node_label, int v) {
    auto it = node_label.find(v);
    return (it == node_label.end() || it->second == -1);
}

static bool is_unlabeled(
    const unordered_map<int,int>& node_label,
    const unordered_map<int, GhostEntry>& ghost_nodes,
    const unordered_map<int, vector<int>>& local_adj,
    int v
) {
    if (local_adj.count(v)) return is_unlabeled_local(node_label, v);
    auto it = ghost_nodes.find(v);
    if (it == ghost_nodes.end()) return true;
    return (it->second.label == -1);
}

static int get_degree(const unordered_map<int,int>& global_degree, int v) {
    auto it = global_degree.find(v);
    return (it != global_degree.end()) ? it->second : -1;
}

static void make_displs(const vector<int>& counts, vector<int>& displs) {
    displs.resize(counts.size());
    int run = 0;
    for (size_t i = 0; i < counts.size(); ++i) { displs[i] = run; run += counts[i]; }
}

static void sort_unique(vector<int>& v) {
    sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
}

static int owner_of(int vid, int nprocs) {
    return (vid % nprocs + nprocs) % nprocs;
}
static bool is_my_partition(int vid, int nprocs, int procId){
    return (vid % nprocs) == procId;
}

static int count_local_labeled(unordered_map<int,int> node_label){
    int c=0;
    for (auto &kv: node_label){
        if (kv.second>=0) ++c;
    }
    return c;
};

static void recompute_partition_sizes(
    int numParts,
    const unordered_map<int, vector<int>>& local_adj,
    const unordered_map<int, int>& node_label,
    vector<int>& part_size_global
) {
    if ((int)part_size_global.size() != numParts) {
        part_size_global.assign(numParts, 0);
    }
    vector<int> local_count(numParts, 0);
    for (const auto& kv : local_adj) {
        int u = kv.first;
        auto it = node_label.find(u);
        if (it != node_label.end() && it->second >= 0) {
            int p = it->second;
            if (0 <= p && p < numParts) ++local_count[p];
        }
    }
    MPI_Allreduce(local_count.data(), part_size_global.data(),
                  numParts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

static void fetch_ghost_nodes(
    const unordered_set<int> &to_request_ghosts,
    const unordered_map<int, vector<int>> &local_adj,
    const unordered_map<int, int> &node_label,
    int procId,
    int nprocs,
    unordered_map<int, GhostEntry> &ghost_nodes
) {
    vector<vector<int>> requests_per_rank(nprocs);
    for (int ghost : to_request_ghosts) {
        int owner = owner_of(ghost, nprocs);
        requests_per_rank[owner].push_back(ghost);
    }

    vector<int> send_counts(nprocs, 0), recv_counts(nprocs, 0);
    for (int r = 0; r < nprocs; ++r) send_counts[r] = (int)requests_per_rank[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> sdispls(nprocs, 0), rdispls(nprocs, 0);
    int s_total = 0, r_total = 0;
    for (int r = 0; r < nprocs; ++r) { sdispls[r] = s_total; s_total += send_counts[r]; }
    for (int r = 0; r < nprocs; ++r) { rdispls[r] = r_total; r_total += recv_counts[r]; }

    vector<int> sendbuf(s_total), recvbuf(r_total);
    for (int r = 0, idx = 0; r < nprocs; ++r)
        for (int g : requests_per_rank[r]) sendbuf[idx++] = g;

    MPI_Alltoallv(sendbuf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                  recvbuf.data(), recv_counts.data(), rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

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

    vector<int> recv_reply_counts(nprocs, 0);
    MPI_Alltoall(send_reply_counts.data(), 1, MPI_INT,
                 recv_reply_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

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

    for (int i = 0; i < recv_reply_total; ) {
        int node_id = recv_reply[i++];
        int degree  = recv_reply[i++];
        int label   = recv_reply[i++];

        vector<int> nbrs;
        nbrs.reserve(degree);
        for (int d = 0; d < degree; ++d) nbrs.push_back(recv_reply[i++]);

        ghost_nodes[node_id] = GhostEntry{ move(nbrs), label };
    }
}

void build_local_graph_and_ghosts(
    int rank,
    int nprocs,
    const unordered_map<int, vector<int>>& local_adj,
    const unordered_map<int, int>& node_label,
    const unordered_map<int, GhostEntry>& ghost_nodes,
    Graph& local_partition_graphs,
    GhostNodes& local_partition_ghosts
) {

    vector<int> locals;
    locals.reserve(local_adj.size());
    for (const auto& kv : local_adj) locals.push_back(kv.first);
    sort_unique(locals);

    unordered_map<int,int> local_g2l;
    local_g2l.reserve(locals.size()*2);
    for (int i = 0; i < (int)locals.size(); ++i) local_g2l[locals[i]] = i;

    const int L = (int)locals.size();
    local_partition_graphs.num_vertices = L;
    local_partition_graphs.global_ids.resize(L);
    local_partition_graphs.vertex_labels.resize(L);
    for (int i = 0; i < L; ++i) {
        const int g = locals[i];
        local_partition_graphs.global_ids[i] = g;
        auto it = node_label.find(g);
        local_partition_graphs.vertex_labels[i] = (it == node_label.end() ? -1 : it->second);
    }

    local_partition_graphs.row_ptr.clear();
    local_partition_graphs.row_ptr.resize(L + 1, 0);
    local_partition_graphs.col_indices.clear();
    local_partition_graphs.col_indices.reserve(
        accumulate(locals.begin(), locals.end(), 0,
            [&](int acc, int u){ 
                auto it = local_adj.find(u);
                return acc + (it == local_adj.end() ? 0 : (int)it->second.size());
            })
    );

    unordered_set<int> ghosts_unknown; 
    ghosts_unknown.reserve(1024);

    const int ghost_base = L;
    for (int li = 0; li < L; ++li) {
        const int u_global = locals[li];

        auto it = local_adj.find(u_global);
        vector<int> nbrs = (it == local_adj.end()) ? vector<int>() : it->second;
        sort_unique(nbrs);

        local_partition_graphs.row_ptr[li] = (int)local_partition_graphs.col_indices.size();

        for (int v_global : nbrs) {
            const int own = owner_of(v_global, nprocs);
            if (own == rank) {
                auto jt = local_g2l.find(v_global);
                local_partition_graphs.col_indices.push_back(jt->second);
            }
            else {
                auto kt = local_partition_ghosts.global_to_local.find(v_global);
                if (kt == local_partition_ghosts.global_to_local.end()) {
                    const int new_idx = (int)local_partition_ghosts.global_ids.size();
                    local_partition_ghosts.global_to_local[v_global] = new_idx;
                    local_partition_ghosts.global_ids.push_back(v_global);

                    int glabel = -1;
                    auto gt = ghost_nodes.find(v_global);
                    if (gt != ghost_nodes.end()) glabel = gt->second.label;
                    local_partition_ghosts.ghost_labels.push_back(glabel);
                    if (glabel == -1) ghosts_unknown.insert(v_global);
                    local_partition_graphs.col_indices.push_back(ghost_base + new_idx);
                } else {
                    local_partition_graphs.col_indices.push_back(ghost_base + kt->second);
                }
            }
        }
    }
    local_partition_graphs.row_ptr[L] = (int)local_partition_graphs.col_indices.size();
    local_partition_graphs.num_edges = (int)local_partition_graphs.col_indices.size();

    vector<vector<int>> requests(nprocs);
    for (int gid : ghosts_unknown) {
        const int own = owner_of(gid, nprocs);
        requests[own].push_back(gid);
    }
    for (auto& v : requests) { if (!v.empty()) sort_unique(v); }

    vector<int> sendcounts(nprocs, 0);
    for (int p = 0; p < nprocs; ++p) sendcounts[p] = (int)requests[p].size();
    vector<int> sdispls; make_displs(sendcounts, sdispls);

    vector<int> recvcounts(nprocs, 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    vector<int> rdispls; make_displs(recvcounts, rdispls);

    const int sendN = accumulate(sendcounts.begin(), sendcounts.end(), 0);
    const int recvN = accumulate(recvcounts.begin(), recvcounts.end(), 0);

    vector<int> sendbuf(sendN);
    for (int p = 0; p < nprocs; ++p) {
        copy(requests[p].begin(), requests[p].end(), sendbuf.begin() + sdispls[p]);
    }
    vector<int> recvbuf(recvN);

    MPI_Alltoallv(
        sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_INT,
        recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_INT,
        MPI_COMM_WORLD
    );

    vector<int> reply_labels(recvN, -1);
    for (int i = 0; i < recvN; ++i) {
        int gid = recvbuf[i];
        auto it = node_label.find(gid);
        reply_labels[i] = (it == node_label.end() ? -1 : it->second);
    }

    vector<int> recv_labels(sendN, -1);
    MPI_Alltoallv(
        reply_labels.data(), recvcounts.data(), rdispls.data(), MPI_INT,
        recv_labels.data(),  sendcounts.data(), sdispls.data(), MPI_INT,
        MPI_COMM_WORLD
    );

    for (int i = 0; i < sendN; ++i) {
        int gid = sendbuf[i];
        int label = recv_labels[i];
        auto kt = local_partition_ghosts.global_to_local.find(gid);
        if (kt != local_partition_ghosts.global_to_local.end()) {
            int loc = kt->second;
            if (local_partition_ghosts.ghost_labels[loc] == -1) {
                local_partition_ghosts.ghost_labels[loc] = label;
            }
        }
    }
}


// ======== Main ========
void partition_expansion(
    int procId,
    int nprocs,
    int numParts,
    int theta,
    const vector<int> &seeds,
    const unordered_map<int, int> &global_degree,
    const unordered_map<int, vector<int>> &local_adj,
    Graph &local_partition_graphs,
    GhostNodes &local_partition_ghosts,
    const bool verbose
) {
    vector<int> part_size_global(numParts, 0);
    vector<queue<int>> frontiers(numParts);
    vector<int> my_parts;

    unordered_map<int,int> node_label;
    unordered_map<int, GhostEntry> ghost_nodes;


    for (auto &kv : local_adj) node_label[kv.first] = -1;

    unordered_set<int> initial_ghost_seed;
    for (int p=0; p<numParts; ++p) {
        int seed = seeds[p];
        if (is_my_partition(p, nprocs, procId)) {
            my_parts.push_back(p);
            frontiers[p].push(seed);
            if (local_adj.count(seed)) {
                node_label[seed] = p;
                if (verbose) cout << "[Rank " << procId << "] seed local P" << p << " <- " << seed << "\n";
            } else {
                initial_ghost_seed.insert(seed);
                if (verbose) cout << "[Rank " << procId << "] seed ghost P" << p << " <- " << seed << "\n";
            }
        }
    }

    fetch_ghost_nodes(initial_ghost_seed, local_adj, node_label, procId, nprocs, ghost_nodes);
    recompute_partition_sizes(numParts, local_adj, node_label, part_size_global);

    int total_num = (int)global_degree.size();

    int total_labeled=0;
    int local_labeled=count_local_labeled(node_label);
    MPI_Allreduce(&local_labeled, &total_labeled, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int round = 0;
    while (total_labeled < total_num) {
        if (verbose) cout << "[Rank " << procId << "] ===== Round " << round << " =====\n";

        vector<vector<int>> newly_labeled_local(numParts), newly_labeled_ghost(numParts);

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
                    if (is_my_partition(best, nprocs, procId)) {
                        frontiers[best].push(v);
                    }
                }
            }
        }

        size_t frontier_total = 0;
        for (int p : my_parts) frontier_total += frontiers[p].size();
        if (verbose) cout << "[Rank " << procId << "] round " << round << " frontier_total=" << frontier_total << "\n";

        struct ExpandTask { int p; int u; int deg; };
        vector<ExpandTask> expandables;
        expandables.reserve(frontier_total);

        for (int p : my_parts) {
            queue<int> q = frontiers[p];
            while (!q.empty()) {
                int u = q.front(); q.pop();
                expandables.push_back({p, u, get_degree(global_degree,u)});
            }
        }

        sort(expandables.begin(), expandables.end(),
                  [](const ExpandTask& a, const ExpandTask& b){
                      if (a.deg != b.deg) return a.deg < b.deg;
                      if (a.p   != b.p  ) return a.p   < b.p;
                      return a.u < b.u;
                  });

        int budget = max(0, theta);
        int K = min((int)expandables.size(), budget);

        vector<queue<int>> carry_frontiers(numParts);
        unordered_map<int, unordered_set<int>> selected_by_part;
        selected_by_part.reserve(my_parts.size());

        for (int i = 0; i < K; ++i) {
            selected_by_part[expandables[i].p].insert(expandables[i].u);
        }
        for (int p : my_parts) {
            queue<int> q = frontiers[p];
            while (!q.empty()) {
                int u = q.front(); q.pop();
                if (selected_by_part[p].count(u) == 0) {
                    carry_frontiers[p].push(u);
                }
            }
        }
        if (verbose) {
            cout << "[Rank " << procId << "] round " << round
                      << " expand_selected=" << K
                      << " (theta=" << theta << ")\n";
        }

        unordered_set<int> to_request_ghosts;
        unordered_set<int> visited;
        vector<queue<int>> next_frontiers(numParts);

        for (int p : my_parts) {
            queue<int> q = frontiers[p];
            while (!q.empty()) {
                int u = q.front(); q.pop();
                const vector<int>* nbrs = nullptr;
                if (local_adj.count(u)) nbrs=&local_adj.at(u);
                else if (ghost_nodes.count(u)) nbrs=&ghost_nodes[u].nbrs;
                if (!nbrs) continue;

                for (int v : *nbrs) {
                    if (!local_adj.count(v)) {
                        auto git = ghost_nodes.find(v);
                        if (git==ghost_nodes.end() || git->second.label==-1) to_request_ghosts.insert(v);
                    }
                    if (is_unlabeled(node_label, ghost_nodes, local_adj, v) && visited.insert(v).second) {
                        next_frontiers[p].push(v);
                    }
                }
            }
        }
        if (verbose) cout << "[Rank " << procId << "] round " << round << " to_request_ghosts=" << to_request_ghosts.size() << "\n";

        fetch_ghost_nodes(to_request_ghosts, local_adj, node_label, procId, nprocs, ghost_nodes);

        vector<vector<int>> req_per_owner(nprocs);
        for (int p=0; p<numParts; ++p) {
            while (!next_frontiers[p].empty()) {
                int v = next_frontiers[p].front(); next_frontiers[p].pop();
                if (!is_unlabeled(node_label, ghost_nodes, local_adj, v)) continue;
                int ow = owner_of(v, nprocs);
                req_per_owner[ow].push_back(v);
                req_per_owner[ow].push_back(1);
                req_per_owner[ow].push_back(p);
            }
        }

        vector<int> send_counts(nprocs,0), recv_counts(nprocs,0);
        for (int r=0;r<nprocs;++r) send_counts[r] = (int)req_per_owner[r].size();
        MPI_Alltoall(send_counts.data(),1,MPI_INT, recv_counts.data(),1,MPI_INT, MPI_COMM_WORLD);

        int recv_req_total = 0; for (int r=0;r<nprocs;++r) recv_req_total += recv_counts[r];
        if (verbose) cout << "[Rank " << procId << "] round " << round << " recv_counts(req)_sum=" << recv_req_total << "\n";

        vector<int> sdis(nprocs,0), rdis(nprocs,0);
        int st=0, rt=0;
        for (int r=0;r<nprocs;++r){ sdis[r]=st; st+=send_counts[r]; }
        for (int r=0;r<nprocs;++r){ rdis[r]=rt; rt+=recv_counts[r]; }
        vector<int> sendbuf(st), recvbuf(rt);
        for (int r=0, idx=0; r<nprocs; ++r) for (int x: req_per_owner[r]) sendbuf[idx++]=x;

        MPI_Alltoallv(sendbuf.data(), send_counts.data(), sdis.data(), MPI_INT,
                      recvbuf.data(), recv_counts.data(), rdis.data(), MPI_INT, MPI_COMM_WORLD);

        unordered_map<int, vector<int>> plist_by_v;
        unordered_map<int, vector<int>> requesters_by_v;
        for (int r=0; r<nprocs; ++r) {
            int start = rdis[r], end = rdis[r]+recv_counts[r];
            for (int i=start; i<end; ) {
                int v   = recvbuf[i++];
                int len = recvbuf[i++];
                if (owner_of(v, nprocs) != procId) { i += len; continue; }
                auto &plist = plist_by_v[v];
                auto &rq    = requesters_by_v[v];
                for (int k=0; k<len; ++k) {
                    int p = recvbuf[i++];
                    plist.push_back(p);
                    rq.push_back(r);
                }
            }
        }

        vector<vector<int>> upd_per_rank(nprocs);
        int owned_claims = 0;
        for (auto &kv : plist_by_v) {
            int v = kv.first;
            auto &plist = kv.second;
            if (!local_adj.count(v)) continue;
            int winner = plist[0];
            for (int p : plist) {
                if (part_size_global[p] < part_size_global[winner] ||
                   (part_size_global[p]==part_size_global[winner] && p < winner)) winner = p;
            }
            node_label[v] = winner;
            if (is_my_partition(winner, nprocs, procId)) {
                newly_labeled_local[winner].push_back(v);
            }
            auto &rq = requesters_by_v[v];
            for (int rr : rq) { upd_per_rank[rr].push_back(v); upd_per_rank[rr].push_back(winner); }
            upd_per_rank[procId].push_back(v); upd_per_rank[procId].push_back(winner);
            ++owned_claims;
        }

        vector<int> u_send_counts(nprocs,0), u_recv_counts(nprocs,0);
        for (int r=0;r<nprocs;++r) u_send_counts[r]=(int)upd_per_rank[r].size();
        MPI_Alltoall(u_send_counts.data(),1,MPI_INT, u_recv_counts.data(),1,MPI_INT, MPI_COMM_WORLD);

        vector<int> u_sdis(nprocs,0), u_rdis(nprocs,0);
        int ust=0, urt=0;
        for (int r=0;r<nprocs;++r){ u_sdis[r]=ust; ust+=u_send_counts[r]; }
        for (int r=0;r<nprocs;++r){ u_rdis[r]=urt; urt+=u_recv_counts[r]; }
        vector<int> u_sendbuf(ust), u_recvbuf(urt);
        for (int r=0, idx=0; r<nprocs; ++r) for (int x: upd_per_rank[r]) u_sendbuf[idx++]=x;

        MPI_Alltoallv(u_sendbuf.data(), u_send_counts.data(), u_sdis.data(), MPI_INT,
                      u_recvbuf.data(), u_recv_counts.data(), u_rdis.data(), MPI_INT, MPI_COMM_WORLD);

        int upd_recv_pairs = urt / 2;

        for (int i=0; i<urt; ) {
            int v      = u_recvbuf[i++];
            int winner = u_recvbuf[i++];
            if (!local_adj.count(v)) {
                auto &ge = ghost_nodes[v];
                ge.label = winner;
                if (is_my_partition(winner, nprocs, procId)) {
                    newly_labeled_ghost[winner].push_back(v);
                }
            }
        }

        if (verbose) {
            cout << "[Rank " << procId << "] round " << round
                      << " owned_claims=" << owned_claims
                      << " upd_recv=" << upd_recv_pairs << "\n";
        }

        local_labeled = count_local_labeled(node_label);
        MPI_Allreduce(&local_labeled, &total_labeled, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        recompute_partition_sizes(numParts, local_adj, node_label, part_size_global);

        for (int p=0; p<numParts; ++p) {
            queue<int> q;
            while (!carry_frontiers[p].empty()) {
                q.push(carry_frontiers[p].front());
                carry_frontiers[p].pop();
            }
            for (int u : newly_labeled_local[p])  q.push(u);
            for (int u : newly_labeled_ghost[p])  q.push(u);
            frontiers[p].swap(q);
        }
        size_t next_frontier_total = 0;
        for (int p=0; p<numParts; ++p) next_frontier_total += frontiers[p].size();
        if (verbose) cout << "[Rank " << procId << "] round " << round
                               << " next_frontier_total=" << next_frontier_total << "\n";

        ++round;
    }

    build_local_graph_and_ghosts(
        procId, nprocs,
        local_adj,
        node_label,
        ghost_nodes,
        local_partition_graphs,
        local_partition_ghosts
    );

    if (verbose) {
        cout << "=== [Rank " << procId << "] CSR Summary ===\n";
        cout << "Vertices: " << local_partition_graphs.num_vertices << "\n";
        cout << "Edges:    " << local_partition_graphs.num_edges << "\n";
        cout << "Ghosts:   " << local_partition_ghosts.global_ids.size() << "\n";
    }
}
