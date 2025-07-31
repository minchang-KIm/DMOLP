#ifndef DMOLP_REPORT_UTILS_H
#define DMOLP_REPORT_UTILS_H

#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include "graph_types.h"

// Phase 2 최종 결과 및 Phase 1/2 비교 출력
void printFinalResults(
    int mpi_rank,
    int current_edge_cut_,
    const std::vector<PartitionInfo>& PI_,
    int num_partitions_,
    long execution_time_ms,
    const Phase1Metrics& phase1_metrics_);

#endif // DMOLP_REPORT_UTILS_H
