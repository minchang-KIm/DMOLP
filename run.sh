#!/bin/bash

# 176번 노드에서 실행파일을 강제로 삭제 후 복사 (Text file busy 방지)
ssh 210.107.197.176 'rm -f /home/intern_graph/intern/Label_Propagation/DMOLP/build_gpu/dmolp_gpu'
scp ./build_gpu/dmolp_gpu 210.107.197.176:/home/intern_graph/intern/Label_Propagation/DMOLP/build_gpu/dmolp_gpu
