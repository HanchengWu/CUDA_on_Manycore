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
  for affinity in 1 2 3 4; do
    for i in `seq 1 5`; do
      echo "${i} ./bfs ${affinity} 1 /home/hwu16/bfs_data/graph${data_num}k.txt"
      eval "./bfs ${affinity} 1 /home/hwu16/bfs_data/graph${data_num}k.txt"
    done
  done 
done

# M datasets
for data_num in 1 2 4 8 16; do
  for affinity in 1 2 3 4; do
    for i in `seq 1 5`; do
      echo "${i} ./bfs ${affinity} 1 /home/hwu16/bfs_data/graph${data_num}M.txt"
      eval "./bfs ${affinity} 1 /home/hwu16/bfs_data/graph${data_num}M.txt"
    done
  done 
done

