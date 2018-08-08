#!/bin/bash

#usage 
#display_usage() { 
#  echo "Automating checking testcases for fp" 
#  echo -e "\nUsage: $0 <# int tests> <# int extra tests> <# fp extra tests>\n" 
#} 
#
## if less than two arguments supplied, display usage 
#if [  $# -le 2 ] 
#then 
#  display_usage
#  exit 1
#fi 

# Recompile testcases
make clean; make -j

# Pure integer testcases
#for i in `seq 1 $1`;
#do
#  echo "./bin/testcase$i > out$i.txt; diff out$i.txt testcases/testcase$i.out; rm -rf out$i.txt"
#  eval "./bin/testcase$i > out$i.txt; diff out$i.txt testcases/testcase$i.out; rm -rf out$i.txt"
#done

for filename in data/*.txt; do
  #for ((i=0; i<=3; i++)); do

  echo "./bfs $filename"
  eval "./bfs $filename"
  #eval "mv result.txt data/results/result_$(basename "$filename" .txt).out" # save the results

  echo "diff result.txt data/results/result_$(basename "$filename" .txt).out"
  eval "diff result.txt data/results/result_$(basename "$filename" .txt).out" # compare the output

  #echo "./bfs $filename > out_$(basename "$filename" .txt).out"
  #eval "./bfs $filename > $1/results_$(basename "$filename" .txt).out"
  #echo "./bfs $filename > results_$(basename "$filename" .txt).out; diff $1/results_$(basename "$filename" .txt).out results_$(basename "$filename" .txt).out"
  #  ./MyProgram.exe "$filename" "Logs/$(basename "$filename" .txt)_Log$i.txt"     
  #done 
done
