# DMOLP - Distributed Multi-Objective Label Propagation

분산 다목적 라벨 전파 기반 그래프 파티셔닝 시스템

## 개요

DMOLP는 대규모 그래프를 효율적으로 분할하기 위한 MPI 기반 분산 시스템입니다. 2단계 접근법을 사용하여 최적의 그래프 파티셔닝을 수행합니다.

### 주요 특징

- **Phase 1**: 초기 그래프 로딩 및 해시 기반 분할
- **Phase 2**: 7단계 동적 라벨 전파 알고리즘
- **MPI 분산 처리**: 다중 노드 병렬 실행
- **CUDA 가속**: GPU를 활용한 고성능 계산
- **OpenMP 최적화**: 멀티스레드 병렬 처리

### 성능

- **Edge-cut 개선**: 94%+ 감소
- **실행 시간**: 대규모 그래프 3-4초 내 처리
- **확장성**: MPI를 통한 다중 노드 확장 가능

## 빌드 요구사항

### 필수 의존성

- **CMake** >= 3.18
- **CUDA** >= 11.0
- **MPI** (OpenMPI 또는 MPICH)
- **OpenMP**
- **C++17** 지원 컴파일러

### 선택적 의존성

- **NVIDIA GPU** (CUDA 가속용)
- **Slurm** (클러스터 환경용)

## 빌드 방법

### CPU 버전 (기본)
```bash
# 프로젝트 클론
git clone <repository-url>
cd DMOLP

# 빌드 디렉토리 생성
mkdir build && cd build

# CMake 구성 (CPU 전용)
cmake -DUSE_CUDA=OFF ..

# 컴파일
make -j$(nproc)
```

### GPU 버전 (CUDA 가속)
```bash
# 빌드 디렉토리 생성
mkdir build && cd build

# CMake 구성 (CUDA 활성화)
cmake -DUSE_CUDA=ON ..

# 컴파일
make -j$(nproc)
```

### 빌드 옵션
- `-DUSE_CUDA=ON/OFF`: CUDA 가속 활성화/비활성화
- `-DCMAKE_BUILD_TYPE=Release/Debug`: 빌드 타입 설정

## 실행 방법

### 기본 실행

```bash
# MPI 실행 (2개 프로세스, 4개 파티션)
mpirun -np 2 ./dmolp <graph_file> <num_partitions>

# 예시
mpirun -np 2 ./dmolp ../data/graph.mtx 4
```

### 지원 그래프 형식

- **METIS 형식** (.graph, .mtx)
- **Adjacency List 형식** (.adj)

## 알고리즘 구조

### Phase 1: 초기 분할
1. 그래프 파일 로딩
2. METIS/ADJ 형식 파싱
3. 해시 기반 초기 분할
4. MPI 노드 간 분산

### Phase 2: 7단계 최적화
1. **RV/RE 계산**: 정점/간선 비율 계산
2. **불균형 계산**: 파티션 간 불균형 측정
3. **Edge-cut 계산**: 경계 정점 추출
4. **동적 라벨 전파**: 스코어 기반 라벨 업데이트
5. **파티션 업데이트**: MPI 통신으로 정보 교환
6. **수렴 확인**: 알고리즘 수렴 조건 검사
7. **다음 반복 준비**: 상태 업데이트

## 성능 메트릭

- **Edge-cut**: 파티션 간 잘린 간선 수 (최소화 목표)
- **Vertex Balance**: 정점 분산 균형도
- **Edge Balance**: 간선 분산 균형도
- **실행 시간**: 전체 알고리즘 수행 시간

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## 기여하기

1. 이슈 생성
2. 기능 브랜치 생성
3. 변경사항 커밋
4. 풀 리퀘스트 제출

## 문의

버그 리포트나 기능 요청은 [Issues](../../issues)에 등록해주세요.
