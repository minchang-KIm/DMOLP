---
applyTo: '/home/intern_graph/intern/Label_Propagation/DMOLP/*'
---
no simulation code
no 임시코드

[배열 설명]
BV (Boundary Vertex) : 파티션 내의 Vertex 중 Remote Edge가 존재하는 Vertex 배열 (Adjacency List 기반 = 이웃노드도 알아야함)
NV (Neighbor Vertex) :BV 배열의 이웃노드 파티션 정보 (Key-Value 기반 = 해당 노드의 파티션 정보만 알면 됨)
PI (Partition info) : 해당 파티션의 정보
PU_RO (Partition Update Received from Own partition) : iteration이 종료되고 자신의 파티션에서 추가되는 Vertex 정보
PU_OV (Partition Update Outgoing Vertex) : iteration이 종료되고 다른 파티션에 보내야 할 Vertex 정보 (Adjacency list 기반)
PU_ON (Partition Update Outgoing Neighbor) : iteration이 종료되고 다른 파티션에 보내야 할 Neighbor 정보 (Key-Value 기반)
PU_RV (Partition Update Received Vertex) : iteration이 종료되고 다른 파티션으로부터 받은 Vertex 정보 (Adjacency list 기반)
PU_RN (Partition Update Received Neighbor) : iteration이 종료되고 다른 파티션으로부터 받은 Neighbor 정보 (Key-Value 기반)
[1step]
CPU에서 각 파티션의 RV(Ratio of Vertex), RE(Ratio of Edge) 값을 계산
[2step]
전체 imbalance값을 계산
[3step]
EC(Edge-cut)값을 계산
계산된 imbalance값과 RV, RE값을 GPU의 PI배열에 전송
CPU memory로부터 BV와 NV를 탐색하여 GPU의 BV와 NV배열에 전송
[4step]
BV 배열에 해당하는 노드에 대하여 PI배열에 저장된 값을 바탕으로 Dynamic Unweighted LP를 수행
이 과정에서 아래 3개의 작업이 이루어짐.
1) LP 과정에서 (((파티션이 변경된 노드의 이웃 중) 기존 파티션에 속해있는 노드) 중 BV배열에 없는 노드)를 PU_RO에 저장.(이는 해당 이웃 노드가 다음 iteration에 자연히 Remote Edge를 하나 포함하기 때문에 RV 배열에 추가해야하기 때문이다.)
2) 파티션이 변경된 노드를 PU_OV에 저장.
3) 파티션이 변경된 노드의 이웃정보를 PU_ON에 저장.
[5step]
BV에 해당하는 모든 노드의 LP가 수행되었다면, PU_OV와 PU_ON의 값을 다른 파티션의 PU_RV와 PU_RN으로 전송
또한, PU_RO에 해당하는 노드를 CPU memory로부터 가져와 BV와 NV로 전송
[6step]
변경된 파티션의 E(Edge-cut)값을 계산
그 후,EC변화량을 계산 (현재 EC - 이전 EC)/(이전 EC)
EC변화량이 ϵ보다 작은 경우가 k번 반복한다면 알고리즘 종료 (ϵ,k는 상수로 우선 각각 0.03, 10으로 설정)
[7step]
1,2,4,5,6 step를 알고리즘이 종료될 때까지 반복


김민창
  오후 1:35
•Constraints of graph Partitioning
1.Edge-cut
•가장 보편적인 Constraints: 서로 다른 파티션에 속한 노드 간의 연결 수를 최소화하는 것을 목표로 함
•ER=EC_prev-EC_curr
•EC_prev는 이전 iteration의 Edge-cut수, EC_curr는 현재 iteration의 Edge-cut수
2.   Vertex Balance, Edge Balance
•각 파티션에 포함된 Vertex와 Edge를 균등하게 유지하는 것을 목표로 함 (분산 시스템의 로드 벨런싱 관련)
•제안기법에서는 아래의 수식을 통해 Vertex Balance와 Edge Balance를 계산
•RV(Ratio of Vertex), RE(Ratio of Edge), V_P: 파티션 P에 속한 노드의 집합, |V|: 전체 노드 수, E_P : 파티션 P에 속한 엣지의 집합, |E|: 전체 엣지 수, k: 총 파티션 수
〖RE〗_P=|E_in |/(|E|/k)
〖RV〗_P=|V_P |/(|V|/k)
1.Weighted Label propagation
•LP 실행 시 Vertex Balance와 Edge Balance를 유지하기 위해 아래의 식을 사용하여 라벨 별 점수를 계산
•Score(L)= |u|·(1+P_L ),       u∈N(v),label(u)=L, -1<P_L<1
•Score(L): 라벨 L의 점수, u: 노드 v의 이웃 중 라벨이 L인 노드, P_L: 라벨L의 페널티 함수이며, 해당 값이 양수일 경우 Score(L)가 기존보다 높게 측정되고 음수일 경우 낮게 측정됨
•페널티 함수: P_L=imb_i (RV)·G_RV (L)+imb_i (RE)·G_RE (L)
• imb_i (RV)와 imb_i (RE): 전체 그래프에서 Vertex balance와 Edge balance의 불균형도, G_RV (L)와 G_RE (L): 평균에서 파티션의 RV, RE값의 차이 (해당 값이 음수일 경우 평균값보다 높은 Vertex, Edge를 가지는 경우로 점수가 낮게 부여되며, 양수일 경우 그 반대)
•Gain 함수: G_RV (L)=(1-〖RV〗_L)/K,     G_RE (L)=(1-〖RE〗_L)/K
2.   Dynamic control of weight
•G_RV (L)와 G_RE (L) (라벨이 𝑳인 파티션의 Gain값)는 iteration마다 해당 값의 중요도가 다를 수 있음 (가중치 보정이 필요)
     (예: Vertex Balance의 불균형도가 높다면 G_RV (L)에 더 높은 점수를 부여)
•iteration마다 다음 수식으로 Vertex Balance와 Edge Balance의 불균형도를 계산
〖imb〗_i (RV)=(Var(RV))/(Var(RV)+Var(RE)),     〖imb〗_i (EB)=(Var(RE))/(Var(RV)+Var(RE))
    (Var(RV)와 Var(RE): iteration i에서의 각 파티션의 RV, RE값들의 분산)
•Evaluation Metric
<Edge-cut> ∑_((u,v)∈E)▒〖δ(p(u)≠p(v))〗
<Vertex/ Edge Balance> VB=(max_i |V_i |)/(|V|/k), EB=(max_i |E_i |)/(|E|/k)
<Execution Time> T_total=T_end -T_start