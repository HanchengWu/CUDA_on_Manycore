#!/bin/bash

#usage 
display_usage() { 
  echo "Benchmark bfs" 
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

#cols
#for cols in 250 500 1000 2000 4000; do
#  for i in `seq 1 25`; do
#    echo "${i} ./pathfinder 200000 ${cols} 20"
#    eval "./pathfinder 200000 ${cols} 20"
#  done
#done

#for height in 10 20 40 80 160 320 640 1280; do
#  for i in `seq 1 25`; do
#    echo "${i} ./pathfinder 200000 2000 ${height}"
#    eval "./pathfinder 200000 2000 ${height}"
#  done
#done

for rows in 10000 20000 40000; do
  for cols in 20 40; do
    for i in `seq 1 10`; do
      echo "${i} ./pathfinder ${rows} ${cols} 20"
      eval "./pathfinder ${rows} ${cols} 20"
    done
  done
done

