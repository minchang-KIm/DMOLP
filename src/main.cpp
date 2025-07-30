
#include <iostream>
#include <vector>
#include <set>
#include <mpi.h>
#include "phase1.h"
#include "types.h"
#include <cuda_runtime.h>
#include <unistd.h>



// 메인 함수
int main(int argc, char* argv[]) {
    // [디버그] main 진입 즉시 host, rank, argc, argv[0] 출력 (노드별 실행 확인용)
    char debug_hostname[256];
    gethostname(debug_hostname, sizeof(debug_hostname));
    std::cout << "[DEBUG] main() 진입: host=" << debug_hostname
              << ", argc=" << argc
              << ", argv[0]=" << (argc > 0 ? argv[0] : "(null)") << std::endl;
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // 각 rank별로 GPU 할당
    int local_rank = 0;
    char* env_local_rank = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (env_local_rank != nullptr) {
        local_rank = std::atoi(env_local_rank);
    }
    cudaSetDevice(local_rank);
    int device = -1;
    cudaGetDevice(&device);
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    std::cout << "[RANK " << mpi_rank << "] host: " << hostname << ", local_rank=" << local_rank << ", cudaGetDevice()=" << device << std::endl;

    if (argc < 3) {
        if (mpi_rank == 0) {
            std::cout << "사용법: mpirun -np <GPU수> -hostfile hosts.txt ./mpi_distributed_workflow_v2 <그래프파일> <파티션수>\n";
            std::cout << "  <GPU수>: hosts.txt의 slots 총합 (예: 2노드×2GPU=4)\n";
            std::cout << "  <파티션수>: 분할할 파티션 개수 (GPU수와 다를 수 있음)\n";
        }
        MPI_Finalize();
        return 1;
    }

    int num_partitions = std::atoi(argv[2]);
    std::string filename = argv[1];

    // 내 rank가 담당할 파티션 id 리스트 구하기 (파티션 수 > rank 수 가능)
    std::vector<int> my_partitions;
    for (int p = 0; p < num_partitions; ++p) {
        if (p % mpi_size == mpi_rank) my_partitions.push_back(p);
    }

    // 각 파티션별로 반복 처리
    for (int part_id : my_partitions) {
        // Phase 1: 그래프 분할 및 분배 (파티션별)
        Graph local_graph;
        std::vector<int> vertex_labels;
        // phase1 분할 및 분배 (part_id만 담당)
        Phase1Metrics phase1_metrics = phase1_partition_and_distribute(part_id, num_partitions, num_partitions, local_graph, vertex_labels, filename);

        // 각 rank의 파티션 분배 결과 출력
        int total_vertices = phase1_metrics.total_vertices;
        int per_part = (total_vertices + num_partitions - 1) / num_partitions;
        int global_start = part_id * per_part;
        extern void printPhase1PartitionSummary(int mpi_rank, int mpi_size, int num_partitions, const Graph& local_graph, const std::vector<int>& vertex_labels, int global_start);
        printPhase1PartitionSummary(mpi_rank, mpi_size, num_partitions, local_graph, vertex_labels, global_start);

        // [디버그] 각 rank가 현재 가진 라벨 ID(고유값)와 노드 수 출력
        std::set<int> unique_labels(vertex_labels.begin(), vertex_labels.end());
        std::cout << "[RANK " << mpi_rank << "] part_id=" << part_id
                  << ", 노드 수=" << local_graph.num_vertices
                  << ", 라벨 ID(고유값)=";
        for (auto it = unique_labels.begin(); it != unique_labels.end(); ++it) {
            if (it != unique_labels.begin()) std::cout << ",";
            std::cout << *it;
        }
        std::cout << std::endl;

        // Phase 2: 7단계 알고리즘 (각 파티션별)
        try {
            // phase2: 7단계 알고리즘을 함수형으로 직접 호출 (클래스 없이)
            extern void dmolp_distributed_workflow_run(int argc, char** argv, const Graph& local_graph, const std::vector<int>& vertex_labels, const Phase1Metrics& phase1_metrics);
            dmolp_distributed_workflow_run(argc, argv, local_graph, vertex_labels, phase1_metrics);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Finalize();
    return 0;
}
