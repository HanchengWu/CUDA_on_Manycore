# C compiler
#CC = g++
ICC = icpc
CC_FLAGS = -g -fopenmp -O3
ICC_FLAGS = -g -qopenmp -O3

all: bfs bfs_offload

bfs: bfs.cpp 
	$(ICC) $(ICC_FLAGS) bfs.cpp -o bfs 

bfs_offload: bfs.cpp
	$(ICC) $(CC_FLAGS)  bfs.cpp -o bfs_offload

clean:
	rm -f bfs bfs_offload result.txt
