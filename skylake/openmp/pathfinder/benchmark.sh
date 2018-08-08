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
      for i in `seq 1 10`; do
        echo "${i} ./pathfinder ${cols} ${rows} 24"
        eval "./pathfinder ${cols} ${rows} 24"
      done
  done
done

