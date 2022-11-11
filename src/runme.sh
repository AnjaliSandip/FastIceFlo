#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

#Optimal PT solver parameter values for the glacier model configuration and the corresponding DoF
damp=0.99;  
relaxation=0.03;
stability=0.9;

# compile the code
nvcc -arch=sm_70 -O3 -lineinfo   ssa_fem_pt.cu   -Ddmp=$damp -Dstab=$stability -Drela=$relaxation 

# run the code
./a.out

#Generate NSIGHT Systems report
#nsys profile --stats=true ./a.out

#Generate NSIGHT Compute report
#ncu --set full --import-source yes -o report ./a.out;

