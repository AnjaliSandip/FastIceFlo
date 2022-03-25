#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh



# compile the code
nvcc -arch=sm_70 -O3 gpu.cu

# run the code
nsys profile -t nvtx ./a.out
nsys profile --stats=true ./a.out
#ncu --set full -o JKS2e4_1iteration ./a.out

