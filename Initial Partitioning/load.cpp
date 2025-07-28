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
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (procId == 0) {
        global_degree.insert(local_degree.begin(), local_degree.end());

        for (int p = 1; p < nprocs; ++p) {
            int len;
            MPI_Status status;
            
            MPI_Recv(&len, 1, MPI_INT, p, 100, MPI_COMM_WORLD, &status);
            
            if (len > 0) {
                vector<int> recvBuffer(len);
                MPI_Recv(recvBuffer.data(), len, MPI_INT, p, 101, MPI_COMM_WORLD, &status);

                for (int i = 0; i < len; i += 2) {
                    int node = recvBuffer[i];
                    int deg = recvBuffer[i + 1];
                    global_degree[node] = deg;
                }
            }
        }
    } else {
        vector<int> sendBuffer;
        for (auto &[node, deg] : local_degree) {
            sendBuffer.push_back(node);
            sendBuffer.push_back(deg);
        }

        int len = sendBuffer.size();
        
        MPI_Send(&len, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
        if (len > 0) {
            MPI_Send(sendBuffer.data(), len, MPI_INT, 0, 101, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}