#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh



# compile the code
#nvcc -arch=sm_70 -O3 ssa_fem_pt.cu
nvcc -arch=sm_70 -O3 -lineinfo ssa_fem_pt.cu
#nvcc  SingleRun.cu

# run the code
#nsys profile -t nvtx  ./a.out
#nsys profile --stats=true  ./a.out
ncu --set full -o JKS7e5report ./a.out;
