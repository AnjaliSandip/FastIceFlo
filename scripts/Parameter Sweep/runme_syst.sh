#!/bin/bash
  
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh


DO_SAVE=true

if [ "$DO_SAVE" = "true" ]; then

     FILE="./output.txt"

     if [ -f "$FILE" ]; then
          echo "Systematic results (file $FILE) already exists. Remove to continue."
          exit 0
     else
          echo "Launching systematics (saving results to $FILE):"

          for damp in $(seq 0.99 0.001 0.999)
          do
               for rele in $(seq 0.99 0.001 0.999)
               do
               echo "Running damp=$damp  relaxation=$rele"
               nvcc -arch=sm_70 -O3 ParameterSweep.cu -Ddmp=$damp -Drela=$rele -DDO_SAVE=$DO_SAVE
               ./a.out
          done
       done
     fi
fi