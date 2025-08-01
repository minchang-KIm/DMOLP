#include "graph_types.h"
#include <iostream>
#include <iomanip>
#include "report_utils.h"

void printComparisonReport(const PartitioningMetrics& m1, const PartitioningMetrics& m2) {
    std::cout << "\n┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    메트릭 비교 결과                         │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Edge-cut:                                                   │\n";
    std::cout << "│   Phase 1 (초기): " << std::setw(10) << m1.edge_cut << "                              │\n";
    std::cout << "│   Phase 2 (최종): " << std::setw(10) << m2.edge_cut << "                              │\n";
    double edge_cut_improvement = (m1.edge_cut > 0) ? (static_cast<double>(m1.edge_cut - m2.edge_cut) / m1.edge_cut) * 100.0 : 0.0;
    std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_cut_improvement << "%                             │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│ Vertex Balance:                                             │\n";
    std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << m1.vertex_balance << "                             │\n";
    std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << m2.vertex_balance << "                             │\n";
    double vertex_balance_improvement = (m1.vertex_balance > 0) ? ((m1.vertex_balance - m2.vertex_balance) / m1.vertex_balance) * 100.0 : 0.0;
    std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << vertex_balance_improvement << "%                             │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│ Edge Balance:                                               │\n";
    std::cout << "│   Phase 1 (초기): " << std::setw(8) << std::fixed << std::setprecision(4) << m1.edge_balance << "                             │\n";
    std::cout << "│   Phase 2 (최종): " << std::setw(8) << std::fixed << std::setprecision(4) << m2.edge_balance << "                             │\n";
    double edge_balance_improvement = (m1.edge_balance > 0) ? ((m1.edge_balance - m2.edge_balance) / m1.edge_balance) * 100.0 : 0.0;
    std::cout << "│   개선율:         " << std::setw(8) << std::fixed << std::setprecision(2) << edge_balance_improvement << "%                             │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│ 실행 시간:                                                  │\n";
    std::cout << "│   Phase 1 (로딩): " << std::setw(6) << m1.loading_time_ms << " ms                          │\n";
    std::cout << "│   Phase 1 (분산): " << std::setw(6) << m1.distribution_time_ms << " ms                          │\n";
    std::cout << "│   Phase 2 (7단계): " << std::setw(5) << m2.loading_time_ms << " ms                          │\n";
    std::cout << "│   총 소요시간:     " << std::setw(5) << (m1.loading_time_ms + m1.distribution_time_ms + m2.loading_time_ms) << " ms                          │\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n";
    std::cout << "\n=== 알고리즘 성능 요약 ===\n";
    if (edge_cut_improvement > 0) {
        std::cout << "✓ Edge-cut " << std::fixed << std::setprecision(1) << edge_cut_improvement << "% 개선 ("
                  << m1.edge_cut << " → " << m2.edge_cut << ")\n";
    } else {
        std::cout << "⚠ Edge-cut " << std::fixed << std::setprecision(1) << -edge_cut_improvement << "% 악화 ("
                  << m1.edge_cut << " → " << m2.edge_cut << ")\n";
    }
    if (vertex_balance_improvement > 0) {
        std::cout << "✓ Vertex Balance " << std::fixed << std::setprecision(1) << vertex_balance_improvement << "% 개선 ("
                  << m1.vertex_balance << " → " << m2.vertex_balance << ")\n";
    } else {
        std::cout << "⚠ Vertex Balance " << std::fixed << std::setprecision(1) << -vertex_balance_improvement << "% 악화 ("
                  << m1.vertex_balance << " → " << m2.vertex_balance << ")\n";
    }
    if (edge_balance_improvement > 0) {
        std::cout << "✓ Edge Balance " << std::fixed << std::setprecision(1) << edge_balance_improvement << "% 개선 ("
                  << m1.edge_balance << " → " << m2.edge_balance << ")\n";
    } else {
        std::cout << "⚠ Edge Balance " << std::fixed << std::setprecision(1) << -edge_balance_improvement << "% 악화 ("
                  << m1.edge_balance << " → " << m2.edge_balance << ")\n";
    }
    std::cout << "총 소요시간: " << (m1.loading_time_ms + m1.distribution_time_ms + m2.loading_time_ms) << " ms\n";
    std::cout << "\n=== 7단계 알고리즘 완료 ===\n";
}
