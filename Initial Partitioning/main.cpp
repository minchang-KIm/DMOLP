#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "load.hpp"
#include "hub.hpp"
#include "utils.hpp"
#include "seed.hpp"
#include "partition.hpp"

using namespace std;

int main(int argc, char **argv) {
    int required = MPI_THREAD_FUNNELED;
    int provided;
    
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required) {
        if (provided == MPI_THREAD_SINGLE) fprintf(stderr, "MPI only provides MPI_THREAD_SINGLE. OpenMP is not safe\n");
        else fprintf(stderr, "MPI does not required thread level\n");

        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int procId, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (procId == 0) fprintf(stderr, "Command Error");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    const char *filename = argv[1];
    const int numParts = atoi(argv[2]);
    const int theta = atoi(argv[3]);

    if (numParts < 2) {
        fprintf(stderr, "Number of partitons must not less than 2");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    unordered_map<int, vector<int>> graph;
    unordered_map<int, int> local_degree;
    unordered_map<int, int> global_degree;
    vector<int> hub_nodes;
    vector<int> landmarks;
    vector<int> seeds;
    vector<vector<int>> partitions;

    double total_start, total_end;
    double find_hub_start, find_hub_end;
    double find_landmark_start, find_landmark_end;
    double find_seed_start, find_seed_end;
    double partitioning_start, partitioning_end;

    total_start = MPI_Wtime();

    load_graph(filename, procId, nprocs, graph, local_degree);
    gather_degrees(local_degree, global_degree, procId, nprocs);

    if (procId == 0) {
        find_hub_start = MPI_Wtime();
        hub_nodes = find_hub_nodes(global_degree);
        find_hub_end = MPI_Wtime();

        find_landmark_start = MPI_Wtime();
        landmarks = find_landmarks(global_degree);
        find_landmark_end = MPI_Wtime();
    }
    
    sync_unordered_map(procId, 0, global_degree);
    sync_vector(procId, 0, hub_nodes);
    sync_vector(procId, 0, landmarks);

    find_seed_start = MPI_Wtime();
    seeds = find_seeds(procId, nprocs, numParts, landmarks, hub_nodes, graph);
    find_seed_end = MPI_Wtime();
    
    sync_vector(procId, 0, seeds);

    partitioning_start = MPI_Wtime();
    partition_expansion(procId, nprocs, numParts, theta, seeds, global_degree, graph, partitions);
    partitioning_end = MPI_Wtime();

    total_end = MPI_Wtime();

    if (procId == 0) {
        cout << "\n[Results]\n";

        cout << "\n[Hub Nodes]\n";
        cout << "Number of hub nodes: " << hub_nodes.size() << "\n";
        for (int i = 0; i < min(10, (int)hub_nodes.size()); ++i) {
            int node = hub_nodes[i];
            int deg = global_degree[node];
            cout << "Node" << node << " (degree: " << deg << ")\n";
        }
        cout << "[Execution Time]\n";
        cout << (find_hub_end - find_hub_start) << "\n";

        cout << "\n[Landmark Nodes]\n";
        cout << "Number of landmark nodes: " << landmarks.size() << "\n";
        for (int i = 0; i < (int)landmarks.size(); ++i) {
            int node = landmarks[i];
            int deg = global_degree[node];
            cout << "Node" << node << " (degree: " << deg << ")\n";
        }
        cout << "[Execution Time]\n";
        cout << (find_landmark_end - find_landmark_start) << "\n";

        cout << "\n[Seed Nodes]\n";
        cout << "Number of seed nodes: " << seeds.size() << "\n";
        for (int i = 0; i < (int)seeds.size(); ++i)
            cout << "Node" << seeds[i] << "\n";
        cout << "[Execution Time]\n";
        cout << (find_seed_end - find_seed_start) << "\n";

        cout <<"\n[Partitioning Results]\n";
        cout << "Number of nodes in the partition 0" << "\n";
        for (int i = 0; i < min(10, (int)partitions[0].size()); ++i)
            cout << "Node" << partitions[0][i] << "\n";
        cout << "[Execution Time]\n";
        cout << (partitioning_end - partitioning_start) << "\n";

        cout << "\n[Total Execution Time]\n";
        cout << (total_end - total_start) << endl;
    }

    MPI_Finalize();
    return 0;
}