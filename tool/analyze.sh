#!/bin/bash
# Usage: ./analyze.sh input_file

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file="$1"

if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

# pv 설치 여부 확인
if ! command -v pv &>/dev/null; then
    echo "[WARN] 'pv' command not found. Progress display will be disabled. Install 'pv' for a better user experience." >&2
    PV_CMD="cat"
else
    PV_CMD="pv -pt -e"
fi

echo "[INFO] Starting graph analysis for: $input_file"
echo "---------------------------------------------------"

# 1. 노드(정점)와 아크(간선) 수 계산
V=$(head -n 1 "$input_file")
if ! [[ "$V" =~ ^[0-9]+$ ]] || [ "$V" -le 0 ]; then
    echo "Error: Invalid vertex count '$V' in header."
    exit 1
fi

# 파일 크기 (진행률 표시용)
file_size=$(stat -c%s "$input_file")

echo "Nodes:             $V"

# 간선 수 계산을 위한 PV 파이프라인
echo "[INFO] Calculating number of edges... "
E=$(tail -n +2 "$input_file" | $PV_CMD -s "$file_size" | awk '{s += NF} END {printf "%.0f", s/2}' | xargs)
if [ -z "$E" ]; then
    E=0
fi

echo "Arcs (Edges):      $E"

# ---
## 주요 통계량 계산

# 2. 평균 차수 (Average Degree)
if (( V > 0 )); then
    average_degree=$(awk "BEGIN {printf \"%.3f\", (2 * $E) / $V}")
    echo "Average Degree:    $average_degree"
else
    echo "Average Degree:    N/A (No nodes)"
fi

# 3. 최대 차수 (Maximum Degree)
# 대용량 파일 처리를 위해 스트리밍 방식으로 최대 차수 계산
echo "[INFO] Calculating maximum degree..."
max_degree=$(tail -n +2 "$input_file" | $PV_CMD -s "$file_size" | awk '{print NF}' | sort -n | tail -n 1)
if [ -z "$max_degree" ]; then
    max_degree=0
fi
echo "Maximum Degree:    $max_degree"

# 4. 고립된 노드 (Dangling/Isolated Nodes)
echo "[INFO] Counting isolated nodes..."
dangling_nodes=$(tail -n +2 "$input_file" | $PV_CMD -s "$file_size" | awk 'NF==0' | wc -l)
dangling_percent=$(awk "BEGIN {if ($V > 0) printf \"%.2f\", ($dangling_nodes / $V) * 100; else print 0}")
echo "Dangling Nodes:    $dangling_nodes ($dangling_percent%)"
echo "---------------------------------------------------"
echo "[INFO] Analysis complete."