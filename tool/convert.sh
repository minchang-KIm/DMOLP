#!/bin/bash
# Usage: ./convert_undirected.sh input_file

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input="$1"

if [ ! -f "$input" ]; then
    echo "Error: File '$input' not found"
    exit 1
fi

# 결과 파일명 생성
base="${input%.graph-txt}"
output="${base}-undirected.adj.txt"
temp_file=$(mktemp)

echo "[INFO] Input file : $input"
echo "[INFO] Output file: $output"

# 정점 수 읽기
V=$(head -n 1 "$input")

if ! [[ "$V" =~ ^[0-9]+$ ]] || [ "$V" -le 0 ]; then
    echo "Error: Invalid vertex count '$V'"
    exit 1
fi

echo "[INFO] Number of vertices: $V"

# 임시 파일로 변환 작업
echo "[INFO] Converting to undirected graph..."

# tail 출력의 크기 계산
tail_size=$(tail -n +2 "$input" | wc -c)

# 변환 및 정렬
{
    tail -n +2 "$input" | pv -pt -e -s "$tail_size" | \
    awk -v V="$V" '
    {
        u = NR - 1  # 0-based indexing
        for (i = 1; i <= NF; i++) {
            v = $i
            # 유효한 정점 번호인지 확인
                if (v >= 0 && v < V && u != v) {
                    print u, v
                    print v, u
                }
        }
    }'
} | sort --parallel=$(nproc) -S 50% -u -k1,1n -k2,2n > "$temp_file"

echo "[INFO] Building adjacency list..."

# 인접 리스트 생성
{
    echo "$V"
    awk -v V="$V" '
    BEGIN {
        for (i = 0; i < V; i++) {
            adj[i] = ""
        }
    }
    {
        u = $1
        v = $2
        if (adj[u] == "") {
            adj[u] = v
        } else {
            adj[u] = adj[u] " " v
        }
    }
    END {
        for (i = 0; i < V; i++) {
            print adj[i]
        }
    }' "$temp_file"
} > "$output"

# 임시 파일 정리
rm -f "$temp_file"

echo "[INFO] Conversion complete."
echo "[INFO] Output saved to: $output"

# 간단한 통계 출력
total_lines=$(wc -l < "$output")
echo "[INFO] Output file has $total_lines lines (1 header + $((total_lines-1)) vertex adjacency lists)"