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

for rows in 10000 20000 40000; do
  #for cols in 2000 4000; do
  for cols in 20 40; do
    for affinity in 1 2 3 4; do
      for i in `seq 1 10`; do
        echo "${i} ./pathfinder ${affinity} 1 ${rows} ${cols} 20"
        eval "./pathfinder ${affinity} 1 ${rows} ${cols} 20"
      done
    done
  done
done

