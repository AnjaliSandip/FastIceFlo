#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

# compile the code
nvcc -arch=sm_70 -O3 -lineinfo   ssa_fem_pt_v2.cu
#nvcc -O3 -lineinfo   ssa_fem_pt_v2.cu

# run the code
./a.out

#Generate NSIGHT Systems report
#nsys profile --stats=true ./a.out

#Generate NSIGHT Compute report
#ncu --set full --import-source yes -o report ./a.out;

