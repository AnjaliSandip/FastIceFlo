#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh



# compile the code
nvcc -arch=sm_70 -O3 -lineinfo memcopy_v3.cu 

# run the code
./a.out
#nsys profile --stats=true ./a.out
#ncu --set full -o report ./a.out
#ncu --set detailed -k PT2_x -o  PT2_xreport ./a.out
