#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

module load cuda

# compile the code
nvcc -arch=sm_80 -O3 ssa_fem_pt.cu

# run the code
./a.out
# nsys profile -t nvtx ./a.out
# nsys profile --stats=true ./a.out
# ncu --set full -o JKS2e4_1iteration ./a.out

