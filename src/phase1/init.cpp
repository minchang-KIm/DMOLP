#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "phase1/init.hpp"

using namespace std;

inline int fast_atoi(const char* &p) {
    int x = 0;
    while (unsigned(*p - '0') <= 9) {
        x = x * 10 + (*p - '0');
        ++p;
    }
    while (*p && unsigned(*p - '0') > 9) ++p;
    return x;
}

void load_graph(const char *filename, int procId, int nprocs, unordered_map<int, vector<int>> &adj, unordered_map<int, int> &local_degree, int &V, uint64_t &E) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Process %d: Error opening file: %s\n", procId, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t BUF_SIZE = 16 * 1024 * 1024;
    char *buf = (char*)malloc(BUF_SIZE);
    setvbuf(fp, buf, _IOFBF, BUF_SIZE);

    char line[64];
    if (!fgets(line, sizeof(line), fp)) {
        cerr << "[proc " << procId << "] Error opening file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    V = atoi(line);

    adj.reserve(V / nprocs * 2);
    local_degree.reserve(V / nprocs * 2);

    string l;
    l.reserve(1024);
    int source = 0;
    char* readBuf = (char*)malloc(BUF_SIZE);
    size_t len;
    string partial;

    while ((len = fread(readBuf, 1, BUF_SIZE, fp)) > 0) {
        size_t start = 0;
        for (size_t i = 0; i < len; ++i) {
            if (readBuf[i] == '\n') {
                size_t segLen = i - start;
                if (!partial.empty()) {
                    partial.append(readBuf + start, segLen);
                    l.swap(partial);
                    partial.clear();
                } else {
                    l.assign(readBuf + start, segLen);
                }

                const char* p = l.c_str();
                if (source % nprocs == procId) {
                    auto &nbrs = adj[source];
                    while (*p) {
                        int neighbor = fast_atoi(p);
                        nbrs.push_back(neighbor);
                    }
                } else {
                    while (*p) fast_atoi(p);
                }
                ++source;

                start = i + 1;
            }
        }
        if (start < len) partial.assign(readBuf + start, len - start);
    }

    free(readBuf);
    fclose(fp);

    int local_V = adj.size();
    uint64_t local_E = 0;

    for (auto &kv : adj) {
        auto &neighbors = kv.second;
        local_degree[kv.first] = (int)neighbors.size();
        local_E += (uint64_t)neighbors.size();
    }

    MPI_Allreduce(&local_E, &E, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_V, &V, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    free(buf);
    E /= 2;
}

void gather_degrees(unordered_map<int, int> &local_degree, unordered_map<int, int> &global_degree, int procId, int nprocs) {
    vector<int> sendbuf;
    sendbuf.reserve(local_degree.size() * 2);
    for (auto &[node, deg] : local_degree) {
        sendbuf.push_back(node);
        sendbuf.push_back(deg);
    }

    int send_size = static_cast<int>(sendbuf.size());

    vector<int> recv_size(nprocs);
    MPI_Allgather(&send_size, 1, MPI_INT, recv_size.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displacement(nprocs, 0);
    for (int i= 1; i < nprocs; ++i)
        displacement[i] = displacement[i - 1] + recv_size[i - 1];

    int total_recv = displacement[nprocs - 1] + recv_size[nprocs - 1];
    vector<int> recvbuf(total_recv);
    MPI_Allgatherv(sendbuf.data(), send_size, MPI_INT, recvbuf.data(), recv_size.data(), displacement.data(), MPI_INT, MPI_COMM_WORLD);

    global_degree.clear();
    global_degree.reserve(total_recv / 2);
    
    for (int p = 0; p < nprocs; ++p) {
        int start = displacement[p];
        int size = recv_size[p];
        int total = start + size;

        for (int i = start; i < total; i+=2) {
            int node = recvbuf[i];
            int deg = recvbuf[i + 1];
            global_degree[node] = deg;
        }
    }
}

vector<int> find_hub_nodes(const unordered_map<int, int> &global_degree) {
    vector<pair<int, int>> sorted_degree(global_degree.begin(), global_degree.end());

    sort(sorted_degree.begin(), sorted_degree.end(), [](const auto &a, const auto &b) {
        return a.second < b.second;
    });

    int N = sorted_degree.size();
    vector<int> hub_nodes;
    if (N < 2) return hub_nodes;

    vector<double> cum_y(N);
    double total_deg = 0.0, sum_deg = 0.0;
    for (int i = 0; i < N; ++i) {
        int deg = sorted_degree[i].second;
        total_deg += deg;
        sum_deg += deg;
        cum_y[i] = sum_deg;
    }

    for (int i = 0; i < N; ++i)
        cum_y[i] /= total_deg;

    double inv_N = 1.0 / N;
    double x1 = N * inv_N;
    double y1 = cum_y[N - 1];
    double x0 = (N - 1) * inv_N;
    double y0 = cum_y[N - 2];
    double slope = (y1 - y0) / (x1 - x0);
    double x_intercept = x1 - y1 / slope;

    for (int i = N - 1; i >= 0; --i) {
        double x = (i + 1) * inv_N;
        if (x > x_intercept) hub_nodes.push_back(sorted_degree[i].first);
        else break;
    }

    return hub_nodes;
}

pair<int, int> find_first_seed(const unordered_map<int, int> &global_degree) {
    pair<int, int> first_seed = {-1, -1};

    for (const auto &[node, deg] : global_degree) {
        if (deg > first_seed.second || (deg == first_seed.second && node < first_seed.second)) first_seed = {node, deg};
    }

    return first_seed;
}