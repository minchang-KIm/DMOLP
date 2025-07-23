# DMOLP 알고리즘 흐름도 및 Phase 2 상세 분석
  
**작성일**: 2025년 7월 22일  
**작성자**: 김민창  
**주제**: DMOLP Phase 2 알고리즘의 7단계 흐름 및 구현 세부사항  
  
---
  
##  1. 전체 시스템 흐름도
  

![](../../../../assets/14d6a557eeea685f93da94d233260b480.png?0.2647994348141447)  
  
---
  
## 📊 2. Phase 2 상세 흐름도
  
### 2.1 7단계 알고리즘 개요
  

![](../../../../assets/14d6a557eeea685f93da94d233260b481.png?0.11759573711131632)  
  
### 2.2 각 단계별 상세 흐름
  
#### Step 1: RV/RE 계산 상세
  

![](../../../../assets/14d6a557eeea685f93da94d233260b482.png?0.39127463923145833)  
  
#### Step 2: 불균형 계산 상세
  

![](../../../../assets/14d6a557eeea685f93da94d233260b483.png?0.7090167721200478)  
  
#### Step 3: Edge-cut 계산 상세
  

![](../../../../assets/14d6a557eeea685f93da94d233260b484.png?0.5828780282752404)  
  
#### Step 4: 동적 라벨 전파 (핵심 알고리즘)
  

```
Error: mermaid CLI is required to be installed.
Check https://github.com/mermaid-js/mermaid-cli for more information.

Error: Command failed: npx -p @mermaid-js/mermaid-cli mmdc --theme default --input /tmp/crossnote-mermaid2025622-2188766-fnd11s.7sjnp.mmd --output /home/intern_graph/assets/14d6a557eeea685f93da94d233260b485.png

Error: Parse error on line 13:
...자적 카운터<br/>atomicAdd(label_changes)]   
-----------------------^
Expecting 'SQE', 'DOUBLECIRCLEEND', 'PE', '-)', 'STADIUMEND', 'SUBROUTINEEND', 'PIPE', 'CYLINDEREND', 'DIAMOND_STOP', 'TAGEND', 'TRAPEND', 'INVTRAPEND', 'UNICODE_TEXT', 'TEXT', 'TAGSTART', got 'PS'
Parser3.parseError (/home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/mermaid/dist/mermaid.js:55774:28)
    at #evaluate (file:///home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/puppeteer-core/lib/esm/puppeteer/cdp/ExecutionContext.js:388:19)
    at async ExecutionContext.evaluate (file:///home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/puppeteer-core/lib/esm/puppeteer/cdp/ExecutionContext.js:275:16)
    at async IsolatedWorld.evaluate (file:///home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/puppeteer-core/lib/esm/puppeteer/cdp/IsolatedWorld.js:97:16)
    at async CdpJSHandle.evaluate (file:///home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/puppeteer-core/lib/esm/puppeteer/api/JSHandle.js:146:20)
    at async CdpElementHandle.evaluate (file:///home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/puppeteer-core/lib/esm/puppeteer/api/ElementHandle.js:340:20)
    at async CdpElementHandle.$eval (file:///home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/puppeteer-core/lib/esm/puppeteer/api/ElementHandle.js:494:24)
    at async CdpFrame.$eval (file:///home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/puppeteer-core/lib/esm/puppeteer/api/Frame.js:450:20)
    at async CdpPage.$eval (file:///home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/puppeteer-core/lib/esm/puppeteer/api/Page.js:450:20)
    at async renderMermaid (file:///home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/@mermaid-js/mermaid-cli/src/index.js:265:22)
    at fromText (/home/intern_graph/.npm/_npx/668c188756b835f3/node_modules/mermaid/dist/mermaid.js:151784:21)


```  

  
#### Step 5: 파티션 업데이트 교환 (MPI 통신)
  

![](../../../../assets/14d6a557eeea685f93da94d233260b486.png?0.05868080268698761)  
  
#### Step 6: 수렴 확인
  

![](../../../../assets/14d6a557eeea685f93da94d233260b487.png?0.6841598267650886)  
  
#### Step 7: 다음 반복 준비
  

![](../../../../assets/14d6a557eeea685f93da94d233260b488.png?0.39837633617658685)  
  
---
  
## ⚡ 3. 성능 최적화 지점
  
### 3.1 병목 지점 분석
  

![](../../../../assets/14d6a557eeea685f93da94d233260b489.png?0.44375623452692325)  
  
### 3.2 최적화 전략
  

![](../../../../assets/14d6a557eeea685f93da94d233260b4810.png?0.946778006953088)  
  
---
  
## 🔬 4. 구현 상세 분석
  
### 4.1 GPU 커널 실행 패턴
  

![](../../../../assets/14d6a557eeea685f93da94d233260b4811.png?0.07860574603106008)  
  
### 4.2 MPI 통신 패턴
  

![](../../../../assets/14d6a557eeea685f93da94d233260b4812.png?0.021056528016739406)  
  
---
  
## 📈 5. 수렴 특성 분석
  
### 5.1 수렴 곡선
  

![](../../../../assets/14d6a557eeea685f93da94d233260b4813.png?0.13933000827751618)  
  
### 5.2 균형도 개선 패턴
  

![](../../../../assets/14d6a557eeea685f93da94d233260b4814.png?0.307648708518349)  
  
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
  