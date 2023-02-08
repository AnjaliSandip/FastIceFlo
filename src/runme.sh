#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

#Optimal PT solver parameter values for the glacier model configuration and the corresponding DoFs. In this case for JKS8e4
damp=0.98;
visc_rela=0.03;
vel_rela=0.99;

# compile the code on a Tesla V100 
nvcc -arch=sm_70 -O3 -lineinfo   ssa.cu  -Ddmp=$damp -Dstability=$vel_rela -Drela=$visc_rela

# run the code
./a.out

#Generate NSIGHT Systems report
#nsys profile --stats=true ./a.out

#Generate NSIGHT Compute report
#ncu --set full --import-source yes -o report ./a.out;


