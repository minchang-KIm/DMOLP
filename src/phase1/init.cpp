#include <mpi.h>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>

#include "phase1/init.hpp"

using namespace std;

void load_graph(const char *filename, int procId, int nprocs, unordered_map<int, vector<int>> &adj, unordered_map<int, int> &local_degree, int &V, int &E) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        fprintf(stderr, "Process %d: Error opening file: %s\n", procId, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ios::sync_with_stdio(false);
    string line;
    
    getline(infile, line);
    V = stoi(line);
    
    int source = 0;
    while (getline(infile, line) && source < V) {
        const char *ptr = line.c_str();
        char *end;
        vector<int> source_neighbors;

        while (*ptr) {
            int neighbor = strtol(ptr, &end, 10);
            if (ptr == end) break;
            ptr = end;

            if (source % nprocs == procId) adj[source].push_back(neighbor);
            if (neighbor % nprocs == procId) adj[neighbor].push_back(source);
        }

        ++source;
    }
    
    infile.close();

    int local_V = adj.size();
    int local_E = 0;

    for (auto &[vertex, neighbors] : adj) {
        sort(neighbors.begin(), neighbors.end());
        neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
        local_degree[vertex] = neighbors.size();
        local_E += neighbors.size();
    }

    MPI_Allreduce(&local_E, &E, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_V, &V, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

void gather_degrees(unordered_map<int, int> &local_degree, unordered_map<int, int> &global_degree, int procId, int nprocs) {
    vector<int> sendbuf;
    for (auto &[node, deg] : local_degree) {
        sendbuf.push_back(node);
        sendbuf.push_back(deg);
    }

    int send_size = sendbuf.size();

    vector<int> recv_size(nprocs);
    MPI_Allgather(&send_size, 1, MPI_INT, recv_size.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displacement(nprocs, 0);
    for (int i= 1; i < nprocs; ++i)
        displacement[i] = displacement[i - 1] + recv_size[i - 1];

    int total_recv = displacement[nprocs - 1] + recv_size[nprocs - 1];
    vector<int> recvbuf(total_recv);
    MPI_Allgatherv(sendbuf.data(), send_size, MPI_INT, recvbuf.data(), recv_size.data(), displacement.data(), MPI_INT, MPI_COMM_WORLD);

    global_degree.clear();
    for (int p = 0; p < nprocs; ++p) {
        int start = displacement[p];
        int size = recv_size[p];
        int total = start + size;

        for (int i = 0; i < total; i+=2) {
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

pair<int, int> find_first_seed(const unordered_map<int, int> &global_degree) {
    vector<pair<int, int>> sorted_degree(global_degree.begin(), global_degree.end());
    sort(sorted_degree.begin(), sorted_degree.end(), [](const auto &a, const auto &b) {
        return (a.second > b.second) || (a.second == b.second && a.first < b.first);
    });

    pair<int, int> first_seed;
    if (!global_degree.empty()) first_seed = sorted_degree[0];
    
    return first_seed;
}