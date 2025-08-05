#pragma once
#include "graph_types.h"
#include <vector>
#include <unordered_map>
#include <string>

Phase1Metrics run_phase1(
    int mpi_rank, int mpi_size,
    const char* graph_file,
    int num_parts,
    int theta,
    Graph &local_graph,
    GhostNodes &ghost_nodes
);
