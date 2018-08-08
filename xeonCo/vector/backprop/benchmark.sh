#!/bin/bash

#usage 
display_usage() { 
  echo "Benchmark pathfinder" 
  echo -e "\nUsage: $0\n" 
} 

# if not equal to one arguments supplied, display usage 
if [  $# -ne 0 ] 
then 
  display_usage
  exit 1
fi 

# Recompile testcases
make clean; make -j

for elements in 4096 8192 16384 32768 65536 131072 262144 524288; do
  for affinity in 1 2 3 4; do
    for i in $(seq 25); do 
      echo "${i} ./backprop ${affinity} 1 ${elements}"
      eval "./backprop ${affinity} 1 ${elements}"
    done
  done
done 

