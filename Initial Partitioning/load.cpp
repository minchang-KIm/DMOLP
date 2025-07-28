#include <mpi.h>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "load.hpp"

using namespace std;

void load_graph(const char *filename, int procId, int nprocs, unordered_map<int, vector<int>> &adj, unordered_map<int, int> &local_degree) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        fprintf(stderr, "Process %d: Error opening file: %s\n", procId, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string line;
    int lineNum = -1;
    unordered_map<int, vector<int>> temp_adj;

    while (getline(infile, line)) {
        ++lineNum;
        if (lineNum == 0) continue;

        int source = lineNum - 1;

        istringstream iss(line);
        vector<int> neighbors;
        int neighbor;
        while (iss >> neighbor) {
            neighbors.push_back(neighbor);
        }

        if (source % nprocs == procId) {
            for (int neighbor : neighbors)
                temp_adj[source].push_back(neighbor);
        }

        for (int neighbor : neighbors)
            if (neighbor % nprocs == procId) temp_adj[neighbor].push_back(source);
    }

    infile.close();

    for (auto &[vertex, neighbors] : temp_adj) {
        if (vertex % nprocs == procId) {
            sort(neighbors.begin(), neighbors.end());
            neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());

            adj[vertex] = neighbors;
            local_degree[vertex] = neighbors.size();
        }
    }
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

        for (int i = 0; i < start + size; i+=2) {
            int node = recvbuf[i];
            int deg = recvbuf[i + 1];
            global_degree[node] = deg;
        }
    }
}