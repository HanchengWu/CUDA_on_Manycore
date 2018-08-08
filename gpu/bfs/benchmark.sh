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

# k datasets
for data_num in 1 2 128 256 512; do
  for i in `seq 1 25`; do
    echo "${i} ./bfs ../../data/bfs/graph${data_num}k.txt"
    eval "./bfs ../../data/bfs/graph${data_num}k.txt"
  done
done

# M datasets
for data_num in 1 2 4 8 16; do
  for i in `seq 1 25`; do
    echo "${i} ./bfs ../../data/bfs/graph${data_num}M.txt"
    eval "./bfs ../../data/bfs/graph${data_num}M.txt"
  done
done
