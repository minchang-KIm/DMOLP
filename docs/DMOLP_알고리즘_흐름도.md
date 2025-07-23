# DMOLP 알고리즘 흐름도 및 Phase 2 상세 분석

**작성일**: 2025년 7월 22일  
**작성자**: 김민창  
**주제**: DMOLP Phase 2 알고리즘의 7단계 흐름 및 구현 세부사항  

---

##  1. 전체 시스템 흐름도

```mermaid
flowchart TD
    A[시작: 그래프 파일 입력] --> B[노드별 부분 그래프 할당]
    B --> E[Phase 2 시작: 7단계 반복 알고리즘]
    
    E --> F[Step 1: RV/RE 계산]
    F --> G[Step 2: 불균형 계산]
    G --> H[Step 3: Edge-cut 계산]
    H --> I[Step 4: 동적 라벨 전파]
    I --> J[Step 5: 파티션 업데이트 교환]
    J --> K[Step 6: 수렴 확인]
    K --> L[Step 7: 다음 반복 준비]
    
    L --> M{수렴됨?}
    M -->|No| F
    M -->|Yes| N[최종 결과 수집]
    N --> O[성능 메트릭 계산]
    O --> P[결과 출력 및 종료]
    
    style E fill:#e1f5fe
    style I fill:#ffcdd2
    style J fill:#fff3e0
    style K fill:#e8f5e8
```

---

## 📊 2. Phase 2 상세 흐름도

### 2.1 7단계 알고리즘 개요

```mermaid
flowchart LR
    subgraph "Phase 2: 반복 최적화"
        direction LR
        S1[Step 1<br/>RV/RE 계산<br/>정점/간선 비율] 
        S2[Step 2<br/>불균형 계산<br/>로드 밸런스]
        S3[Step 3<br/>Edge-cut 계산<br/>경계 간선 수]
        S4[Step 4<br/>동적 라벨 전파<br/>GPU 가속 핵심]
        S5[Step 5<br/>파티션 업데이트<br/>MPI 통신]
        S6[Step 6<br/>수렴 확인<br/>종료 조건]
        S7[Step 7<br/>다음 반복 준비<br/>상태 업데이트]
        
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
        S7 -.-> S1
    end
    
    style S4 fill:#ffcdd2
    style S5 fill:#fff3e0
```

### 2.2 각 단계별 상세 흐름

#### Step 1: RV/RE 계산 상세

```mermaid
flowchart LR
    A1[Step 1 시작] --> B1[로컬 정점 수 계산]
    B1 --> C1[로컬 간선 수 계산]
    C1 --> D1[MPI_Allreduce: 전역 정점 수]
    D1 --> E1[MPI_Allreduce: 전역 간선 수]
    E1 --> F1[RV 비율 계산<br/>local_vertices / total_vertices]
    F1 --> G1[RE 비율 계산<br/>local_edges / total_edges]
    G1 --> H1[목표 비율과 비교<br/>target = 1/num_partitions]
    H1 --> I1[Step 1 완료]
    
    style D1 fill:#e3f2fd
    style E1 fill:#e3f2fd
```

#### Step 2: 불균형 계산 상세

```mermaid
flowchart LR
    A2[Step 2 시작] --> B2[각 파티션 크기 수집]
    B2 --> C2[MPI_Allgather: 모든 파티션 정보]
    C2 --> D2[Vertex Balance 계산<br/>max_vertices / avg_vertices]
    D2 --> E2[Edge Balance 계산<br/>max_edges / avg_edges]
    E2 --> F2[불균형 임계값 확인<br/>threshold = 1.1]
    F2 --> G2{임계값 초과?}
    G2 -->|Yes| H2[불균형 플래그 설정]
    G2 -->|No| I2[균형 상태 유지]
    H2 --> J2[Step 2 완료]
    I2 --> J2
    
    style C2 fill:#e3f2fd
    style G2 fill:#fff3e0
```

#### Step 3: Edge-cut 계산 상세

```mermaid
flowchart LR
    A3[Step 3 시작] --> B3[경계 정점 식별]
    B3 --> C3{GPU 사용?}
    C3 -->|Yes| D3[GPU 메모리 데이터 전송]
    C3 -->|No| E3[CPU 멀티스레드 계산]
    
    D3 --> F3[GPU 커널 실행<br/>calculateEdgeCutKernel]
    F3 --> G3[GPU 결과 수집]
    G3 --> H3[로컬 Edge-cut 값]
    
    E3 --> I3[OpenMP 병렬 계산]
    I3 --> H3
    
    H3 --> J3[MPI_Allreduce: 전역 Edge-cut]
    J3 --> K3[이전 값과 비교]
    K3 --> L3[개선률 계산]
    L3 --> M3[Step 3 완료]
    
    style D3 fill:#ffcdd2
    style F3 fill:#ffcdd2
    style J3 fill:#e3f2fd
```

#### Step 4: 동적 라벨 전파 (핵심 알고리즘)
#### [실제 구현] DMOLP GPU 라벨 전파 구조

```mermaid
flowchart LR
    S0["경계 정점 리스트(boundary_vertices) 준비"] --> S1["GPU 메모리 할당 및 데이터 전송"]
    S1 --> S2["CUDA 커널 launch (block/grid)"]
    S2 --> S3["각 boundary vertex별로 1 thread 담당"]
    S3 --> S4["이웃 탐색 및 최적 라벨(best_label) 결정"]
    S4 --> S5{"라벨 변경 발생?"}
    S5 -- Yes --> S6["vertex_labels 갱신 및 atomicAdd(label_changes)"]
    S5 -- No  --> S7["변경 없음"]
    S6 --> S8["모든 thread 완료 후 동기화"]
    S7 --> S8
    S8 --> S9["label_changes 결과를 CPU로 복사"]
    S9 --> S10["GPU 메모리 해제"]

    style S1 fill:#ffcdd2
    style S2 fill:#ffcdd2
    style S3 fill:#ffcdd2
    style S4 fill:#ffcdd2
    style S6 fill:#ffcdd2
```

```mermaid
flowchart LR
    A4["Step 4 시작"] --> B4["경계 정점 리스트 준비"]
    B4 --> C4{"GPU 가속?"}

    C4 -- Yes --> D4["CUDA 메모리 할당 및 데이터 전송"]
    D4 --> E4["GPU 커널 실행 및 라벨 업데이트"]
    E4 --> F4["결과 복사 및 메모리 해제"]
    F4 --> H4["라벨 변경 수 집계"]

    C4 -- No --> N4["OpenMP 병렬화 및 라벨 전파"]
    N4 --> H4

    H4 --> T4["Step 4 완료"]

    style D4 fill:#ffcdd2
    style E4 fill:#ffcdd2
    style N4 fill:#e8f5e8
```

#### Step 5: 파티션 업데이트 교환 (MPI 통신)

```mermaid
flowchart LR
    A5[Step 5 시작] --> B5[로컬 업데이트 수집<br/>PU_OV: Own Vertices]
    B5 --> C5[데이터 크기 확인<br/>MAX_CHUNK_SIZE 체크]
    C5 --> D5{크기 제한 OK?}
    
    D5 -->|Yes| E5[MPI_Allgather: 크기 교환]
    D5 -->|No| F5[경고 출력 및 스킵]
    
    E5 --> G5[수신 버퍼 계산<br/>오프셋 및 크기]
    G5 --> H5[MPI_Allgatherv: 실제 데이터 교환]
    H5 --> I5[PU_RV 구성<br/>Remote Vertices]
    I5 --> J5[이웃 관계 교환<br/>PU_ON → PU_RN]
    J5 --> K5[고스트 노드 업데이트]
    K5 --> L5[Step 5 완료]
    
    F5 --> L5
    
    style E5 fill:#e3f2fd
    style H5 fill:#e3f2fd
    style K5 fill:#fff3e0
```

#### Step 6: 수렴 확인

```mermaid
flowchart LR
    A6[Step 6 시작] --> B6[로컬 메트릭 수집]
    B6 --> C6[라벨 변경 수]
    B6 --> D6[Edge-cut 개선률]
    B6 --> E6[불균형 정도]
    
    C6 --> F6[MPI_Allreduce: 전역 변경 수]
    D6 --> G6[개선률 계산]
    E6 --> H6[균형도 확인]
    
    F6 --> I6{수렴 조건 확인}
    G6 --> I6
    H6 --> I6
    
    I6 --> J6{조건 1:<br/>변경 수 < 임계값?}
    J6 -->|Yes| K6{조건 2:<br/>개선률 < 1%?}
    J6 -->|No| L6[계속 반복]
    
    K6 -->|Yes| M6{조건 3:<br/>최대 반복 도달?}
    K6 -->|No| L6
    
    M6 -->|Yes| N6[수렴 완료]
    M6 -->|No| O6{조건 4:<br/>균형도 안정?}
    
    O6 -->|Yes| N6
    O6 -->|No| L6
    
    N6 --> P6[수렴 플래그 설정]
    L6 --> Q6[반복 계속 플래그]
    
    P6 --> R6[Step 6 완료]
    Q6 --> R6
    
    style F6 fill:#e3f2fd
    style I6 fill:#e8f5e8
    style N6 fill:#c8e6c9
```

#### Step 7: 다음 반복 준비

```mermaid
flowchart LR
    A7[Step 7 시작] --> B7[메트릭 히스토리 업데이트]
    B7 --> C7[이전 값들 저장<br/>edge_cut_history, balance_history]
    C7 --> D7[경계 정점 리스트 갱신]
    D7 --> E7[성능 카운터 업데이트]
    E7 --> F7[메모리 정리<br/>임시 버퍼 해제]
    F7 --> G7[GPU 메모리 동기화]
    G7 --> H7[MPI 동기화 포인트<br/>MPI_Barrier]
    H7 --> I7[반복 카운터 증가]
    I7 --> J7[다음 반복 준비 완료]
    J7 --> K7[Step 7 완료]
    
    style G7 fill:#ffcdd2
    style H7 fill:#e3f2fd
```

---

## ⚡ 3. 성능 최적화 지점

### 3.1 병목 지점 분석

```mermaid
pie title 실행 시간 분포
    "Step 4: 라벨 전파" : 45
    "Step 5: MPI 통신" : 25
    "Step 3: Edge-cut 계산" : 15
    "Step 1-2: 메트릭 계산" : 8
    "Step 6-7: 수렴 및 준비" : 7
```

### 3.2 최적화 전략

```mermaid
flowchart LR
    subgraph "CPU 최적화"
        A1[OpenMP 병렬화]
        A2[SIMD 벡터화]
        A3[캐시 최적화]
        A4[NUMA 인식]
    end
    
    subgraph "GPU 최적화"
        B1[커널 융합]
        B2[메모리 접합]
        B3[공유 메모리]
        B4[다중 스트림]
    end
    
    subgraph "MPI 최적화"
        C1[비동기 통신]
        C2[데이터 압축]
        C3[통신 오버랩]
        C4[토폴로지 인식]
    end
    
    subgraph "알고리즘 최적화"
        D1[적응적 수렴]
        D2[스마트 스케줄링]
        D3[메모리 풀링]
        D4[로드 밸런싱]
    end
```

---

## 🔬 4. 구현 상세 분석

### 4.1 GPU 커널 실행 패턴

```mermaid
gantt
    title GPU 커널 실행 타임라인
    dateFormat X
    axisFormat %s
    
    section Stream 0
    Data Transfer H→D :0, 100
    Kernel Execution  :100, 300
    Data Transfer D→H :300, 350
    
    section Stream 1
    Data Transfer H→D :50, 150
    Kernel Execution  :150, 350
    Data Transfer D→H :350, 400
    
    section Stream 2
    Data Transfer H→D :100, 200
    Kernel Execution  :200, 400
    Data Transfer D→H :400, 450
    
    section Stream 3
    Data Transfer H→D :150, 250
    Kernel Execution  :250, 450
    Data Transfer D→H :450, 500
```

### 4.2 MPI 통신 패턴

```mermaid
sequenceDiagram
    participant P0 as 프로세서 0
    participant P1 as 프로세서 1
    participant P2 as 프로세서 2
    participant P3 as 프로세서 3
    
    Note over P0,P3: Step 5 시작: 파티션 업데이트 교환
    
    par MPI_Allgather (크기 교환)
        P0->>P1: send_count
        P0->>P2: send_count
        P0->>P3: send_count
    and
        P1->>P0: send_count
        P1->>P2: send_count
        P1->>P3: send_count
    and
        P2->>P0: send_count
        P2->>P1: send_count
        P2->>P3: send_count
    and
        P3->>P0: send_count
        P3->>P1: send_count
        P3->>P2: send_count
    end
    
    Note over P0,P3: 버퍼 크기 계산
    
    par MPI_Allgatherv (실제 데이터 교환)
        P0->>P1: actual_data
        P0->>P2: actual_data
        P0->>P3: actual_data
    and
        P1->>P0: actual_data
        P1->>P2: actual_data
        P1->>P3: actual_data
    and
        P2->>P0: actual_data
        P2->>P1: actual_data
        P2->>P3: actual_data
    and
        P3->>P0: actual_data
        P3->>P1: actual_data
        P3->>P2: actual_data
    end
    
    Note over P0,P3: 고스트 노드 업데이트 완료
```

---

## 📈 5. 수렴 특성 분석

### 5.1 수렴 곡선

```mermaid
xychart-beta
    title "Edge-cut 수렴 패턴"
    x-axis "반복 횟수" [0, 5, 10, 15, 20, 25, 30]
    y-axis "Edge-cut 값" 0 --> 100000
    line "Edge-cut" [95000, 75000, 60000, 50000, 42000, 38000, 36500, 35800, 35400, 35200, 35100, 35050, 35020, 35010, 35005]
```

### 5.2 균형도 개선 패턴

```mermaid
xychart-beta
    title "로드 밸런스 개선"
    x-axis "반복 횟수" [0, 5, 10, 15, 20, 25, 30]
    y-axis "균형도" 1.0 --> 2.0
    line "Vertex Balance" [1.85, 1.65, 1.45, 1.32, 1.24, 1.18, 1.14, 1.11, 1.08, 1.06, 1.04, 1.03, 1.02, 1.01, 1.01]
    line "Edge Balance" [1.92, 1.71, 1.52, 1.38, 1.28, 1.21, 1.16, 1.12, 1.09, 1.07, 1.05, 1.04, 1.03, 1.02, 1.01]
```

---

## 🎯 6. 알고리즘 복잡도 분석

### 6.1 시간 복잡도

| 단계 | CPU 복잡도 | GPU 복잡도 | MPI 통신 |
|------|------------|------------|----------|
| Step 1-2 | O(V + E) | - | O(log P) |
| Step 3 | O(E) | O(E/T) | O(log P) |
| Step 4 | O(B·d) | O(B·d/T) | - |
| Step 5 | O(B) | - | O(P·B) |
| Step 6-7 | O(1) | - | O(log P) |

**범례**:
- V: 정점 수
- E: 간선 수  
- B: 경계 정점 수
- d: 평균 차수
- T: GPU 스레드 수
- P: MPI 프로세서 수

### 6.2 공간 복잡도

```
로컬 메모리: O(V/P + E/P)
고스트 노드: O(B)
통신 버퍼: O(B·P)
GPU 메모리: O(V + E) (전체 그래프)
```

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025년 7월 22일  
**다음 리뷰**: 2025년 8월 22일
