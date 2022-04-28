#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh



# compile the code
#nvcc -arch=sm_70 -O3 ssa_fem_pt.cu
nvcc -arch=sm_70 -O3 -lineinfo ssa_fem_pt.cu

#Generic case
# nvcc -arch=sm_70 -O3 -lineinfo ssa_fem_pt.cu -Ddmp=$damp -Drela=$rele 
# damp = 0.1;
# rele = 0.2;

# run the code and generate NVTX decorators
#nsys profile -t nvtx  ./a.out   

#To run the code and generate NSIGHT Systems report
#nsys profile --stats=true  ./a.out   

#To run the code and generate NSIGHT Compute report
ncu --set full -o JKS7e5report ./a.out;


               
