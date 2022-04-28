#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh



# compile the code
damp=0.1
rele=0.2
nvcc -arch=sm_70 -O3 -lineinfo ssa_fem_pt.cu -Ddmp=$damp -Drela=$rele 

#To run the code and generate NSIGHT Systems report
nsys profile --stats=true  ./a.out   

#To run the code and generate NSIGHT Compute report
#ncu --set full -o JKS7e5report ./a.out;


               
