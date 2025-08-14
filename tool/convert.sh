#!/bin/bash
# Usage: ./convert_undirected.sh input_file [memory_limit_gb]

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <input_file> [memory_limit_gb]"
    echo "  memory_limit_gb: Memory limit in GB (default: 2)"
    exit 1
fi

input="$1"
memory_limit_gb="${2:-2}"

if [ ! -f "$input" ]; then
    echo "Error: File '$input' not found"
    exit 1
fi

# 메모리 제한을 MB로 변환
memory_limit_mb=$((memory_limit_gb * 1024))

# 결과 파일명 생성
base="${input%.graph-txt}"
output="${base}-undirected.adj.txt"
temp_dir=$(mktemp -d)
temp_edges="${temp_dir}/edges"
temp_sorted="${temp_dir}/sorted"

# 정리 함수
cleanup() {
    rm -rf "$temp_dir"
}
trap cleanup EXIT

echo "[INFO] Input file : $input"
echo "[INFO] Output file: $output"
echo "[INFO] Memory limit: ${memory_limit_gb}GB"
echo "[INFO] Temp directory: $temp_dir"

# 정점 수 읽기
V=$(head -n 1 "$input")

if ! [[ "$V" =~ ^[0-9]+$ ]] || [ "$V" -le 0 ]; then
    echo "Error: Invalid vertex count '$V'"
    exit 1
fi

echo "[INFO] Number of vertices: $V"

# 파일 크기 체크
file_size=$(stat -c%s "$input")
echo "[INFO] Input file size: $(numfmt --to=iec-i --suffix=B $file_size)"

# 단계 1: directed edges를 undirected edges로 변환 (스트리밍 방식)
echo "[INFO] Step 1: Converting directed to undirected edges..."

# tail 출력의 크기 계산
tail_size=$(tail -n +2 "$input" | wc -c)

# 진행률 표시를 위한 pv 사용 (있는 경우)
if command -v pv >/dev/null 2>&1; then
    progress_cmd="pv -pt -e -s $tail_size"
else
    progress_cmd="cat"
fi

# edges 생성 (메모리 효율적)
tail -n +2 "$input" | $progress_cmd | \
awk -v V="$V" '
{
    u = NR - 1  # 0-based indexing
    for (i = 1; i <= NF; i++) {
        v = $i
        # 유효한 정점 번호인지 확인하고 self-loop 제거
        if (v >= 0 && v < V && u != v) {
            # 더 작은 정점을 먼저 출력하여 중복 제거 최적화
            if (u < v) {
                print u, v
            } else {
                print v, u
            }
        }
    }
}' > "$temp_edges"

edge_count=$(wc -l < "$temp_edges")
echo "[INFO] Generated $edge_count unique edges"

# 단계 2: 정렬 및 중복 제거 (대용량 파일 처리)
echo "[INFO] Step 2: Sorting and removing duplicates..."

# 사용 가능한 CPU 코어 수
num_cores=$(nproc)

# 정렬 메모리 계산 (전체 메모리 제한의 80%)
sort_memory="${memory_limit_mb}M"

# 파일 크기 계산 (진행률 표시용)
edges_size=$(stat -c%s "$temp_edges")

# GNU sort의 대용량 파일 처리 옵션 사용 (진행률 표시와 함께)
if command -v pv >/dev/null 2>&1; then
    echo "[INFO] Sorting $edge_count edges..."
    pv -pt -e -s "$edges_size" "$temp_edges" | \
    sort --parallel="$num_cores" \
         --buffer-size="$sort_memory" \
         --temporary-directory="$temp_dir" \
         -u \
         -k1,1n -k2,2n > "$temp_sorted"
else
    echo "[INFO] Sorting $edge_count edges (no progress bar - install 'pv' for progress display)..."
    sort --parallel="$num_cores" \
         --buffer-size="$sort_memory" \
         --temporary-directory="$temp_dir" \
         -u \
         -k1,1n -k2,2n \
         "$temp_edges" > "$temp_sorted"
fi

sorted_edge_count=$(wc -l < "$temp_sorted")
echo "[INFO] After deduplication: $sorted_edge_count unique edges"

# 단계 3: 인접 리스트 생성 (메모리 효율적)
echo "[INFO] Step 3: Building adjacency list..."

# 배치 크기 계산 (메모리 제한에 따라)
batch_size=$((memory_limit_gb * 100000))
echo "[INFO] Using batch size: $batch_size vertices"

{
    echo "$V"
    
    # 배치별로 인접 리스트 생성
    for ((start=0; start<V; start+=batch_size)); do
        end=$((start + batch_size - 1))
        if [ $end -ge $V ]; then
            end=$((V - 1))
        fi
        
        echo "[INFO] Processing vertices $start to $end..." >&2
        
        # 현재 배치에 해당하는 정점들의 인접 리스트 생성
        awk -v start="$start" -v end="$end" -v V="$V" '
        BEGIN {
            # 배치 범위의 정점들만 초기화
            for (i = start; i <= end; i++) {
                adj[i] = ""
            }
        }
        {
            u = $1
            v = $2
            
            # u가 현재 배치 범위에 있으면 v를 추가
            if (u >= start && u <= end) {
                if (adj[u] == "") {
                    adj[u] = v
                } else {
                    adj[u] = adj[u] " " v
                }
            }
            
            # v가 현재 배치 범위에 있으면 u를 추가
            if (v >= start && v <= end) {
                if (adj[v] == "") {
                    adj[v] = u
                } else {
                    adj[v] = adj[v] " " u
                }
            }
        }
        END {
            # 현재 배치의 결과 출력
            for (i = start; i <= end; i++) {
                print adj[i]
            }
        }' "$temp_sorted"
    done
} > "$output"

echo "[INFO] Conversion complete."
echo "[INFO] Output saved to: $output"

# 통계 출력
output_size=$(stat -c%s "$output")
total_lines=$(wc -l < "$output")

echo "[INFO] === Conversion Statistics ==="
echo "[INFO] Input file size:  $(numfmt --to=iec-i --suffix=B $file_size)"
echo "[INFO] Output file size: $(numfmt --to=iec-i --suffix=B $output_size)"
echo "[INFO] Output lines: $total_lines (1 header + $((total_lines-1)) adjacency lists)"
echo "[INFO] Total unique edges: $sorted_edge_count"

# 간단한 검증
echo "[INFO] === Verification ==="
non_empty_lists=$(tail -n +2 "$output" | grep -c -v '^$' || true)
echo "[INFO] Vertices with neighbors: $non_empty_lists"
echo "[INFO] Vertices without neighbors: $((V - non_empty_lists))"