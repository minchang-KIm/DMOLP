#include <iostream>
#include <iomanip>
#include "report_utils.h"


void printFinalResults(
    int mpi_rank,
    int current_edge_cut_,
    const std::vector<PartitionInfo>& PI_,
    int num_partitions_,
    long execution_time_ms,
    const Phase1Metrics& phase1_metrics_)
{
    if (mpi_rank != 0) return;

    std::cout << "\n=== 최종 결과 (7단계 알고리즘) ===\n";
    std::cout << "Edge-cut: " << current_edge_cut_ << "\n";

    double max_vertex_ratio = 0.0, max_edge_ratio = 0.0;
    for (int i = 0; i < num_partitions_; ++i) {
        max_vertex_ratio = std::max(max_vertex_ratio, PI_[i].RV);
        max_edge_ratio = std::max(max_edge_ratio, PI_[i].RE);
    }
    std::cout << "Vertex Balance (VB): " << max_vertex_ratio << "\n";
    std::cout << "Edge Balance (EB): " << max_edge_ratio << "\n";
    std::cout << "Execution Time: " << execution_time_ms << " ms\n";

    std::cout << "\n파티션별 상세 정보:\n";
    for (int i = 0; i < num_partitions_; ++i) {
        std::cout << "Partition " << i << ": RV=" << PI_[i].RV
                  << ", RE=" << PI_[i].RE
                  << ", P_L=" << PI_[i].P_L
                  << ", G_RV=" << PI_[i].G_RV
                  << ", G_RE=" << PI_[i].G_RE << "\n";
    }

    std::cout << "\n평가 메트릭:\n";
    std::cout << "- Edge-cut: " << current_edge_cut_ << " (최소화 목표)\n";
    std::cout << "- Vertex Balance: " << max_vertex_ratio << " (1.0에 가까울수록 좋음)\n";
    std::cout << "- Edge Balance: " << max_edge_ratio << " (1.0에 가까울수록 좋음)\n";
    std::cout << "- 총 실행시간: " << execution_time_ms << " ms\n";

    std::cout << "\n=== Phase 1 vs Phase 2 (7단계 알고리즘) 비교 ===\n";
    double edge_cut_improvement = 0.0;
    if (phase1_metrics_.initial_edge_cut > 0) {
        edge_cut_improvement = (static_cast<double>(phase1_metrics_.initial_edge_cut - current_edge_cut_) / phase1_metrics_.initial_edge_cut) * 100.0;
    }
    double vertex_balance_improvement = 0.0;
    if (phase1_metrics_.initial_vertex_balance > 0) {
        vertex_balance_improvement = ((phase1_metrics_.initial_vertex_balance - max_vertex_ratio) / phase1_metrics_.initial_vertex_balance) * 100.0;
    }
    double edge_balance_improvement = 0.0;
    if (phase1_metrics_.initial_edge_balance > 0) {
        edge_balance_improvement = ((phase1_metrics_.initial_edge_balance - max_edge_ratio) / phase1_metrics_.initial_edge_balance) * 100.0;
    }
    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    메트릭 비교 결과                         │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Edge-cut:                                                   │\n";
    std::cout << "│   Phase 1 (초기): " << std::setw(10) << phase1_metrics_.initial_edge_cut << "                              │\n";
    std::cout << "│   Phase 2 (최종): " << std::setw(10) << current_edge_cut_ << "                              │\n";
    std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_cut_improvement << "%                             │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│ Vertex Balance:                                             │\n";
    std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << phase1_metrics_.initial_vertex_balance << "                             │\n";
    std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << max_vertex_ratio << "                             │\n";
    std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << vertex_balance_improvement << "%                             │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│ Edge Balance:                                               │\n";
    std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << phase1_metrics_.initial_edge_balance << "                             │\n";
    std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << max_edge_ratio << "                             │\n";
    std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_balance_improvement << "%                             │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│ 실행 시간:                                                  │\n";
    std::cout << "│   Phase 1 (로딩): " << std::setw(6) << phase1_metrics_.loading_time_ms << " ms                          │\n";
    std::cout << "│   Phase 1 (분산): " << std::setw(6) << phase1_metrics_.distribution_time_ms << " ms                          │\n";
    std::cout << "│   Phase 2 (7단계): " << std::setw(5) << execution_time_ms << " ms                          │\n";
    std::cout << "│   총 소요시간:     " << std::setw(5) << (phase1_metrics_.loading_time_ms + phase1_metrics_.distribution_time_ms + execution_time_ms) << " ms                          │\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n";
    std::cout << "\n=== 알고리즘 성능 요약 ===\n";
    if (edge_cut_improvement > 0) {
        std::cout << "✓ Edge-cut " << std::fixed << std::setprecision(1) << edge_cut_improvement << "% 개선 ("
                  << phase1_metrics_.initial_edge_cut << " → " << current_edge_cut_ << ")\n";
    } else {
        std::cout << "⚠ Edge-cut " << std::fixed << std::setprecision(1) << -edge_cut_improvement << "% 악화 ("
                  << phase1_metrics_.initial_edge_cut << " → " << current_edge_cut_ << ")\n";
    }
    if (vertex_balance_improvement > 0) {
        std::cout << "✓ Vertex Balance " << std::fixed << std::setprecision(1) << vertex_balance_improvement << "% 개선\n";
    } else {
        std::cout << "⚠ Vertex Balance " << std::fixed << std::setprecision(1) << -vertex_balance_improvement << "% 악화\n";
    }
    if (edge_balance_improvement > 0) {
        std::cout << "✓ Edge Balance " << std::fixed << std::setprecision(1) << edge_balance_improvement << "% 개선\n";
    } else {
        std::cout << "⚠ Edge Balance " << std::fixed << std::setprecision(1) << -edge_balance_improvement << "% 악화\n";
    }
    std::cout << "\n=== 7단계 알고리즘 완료 ===\n";
}
