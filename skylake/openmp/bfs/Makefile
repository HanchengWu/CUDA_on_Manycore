# C compiler
CC = icpc
CC_FLAGS = -D__PURE_INTEL_C99_HEADERS__  -g -qopenmp -O3

all: bfs

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs 

clean:
	rm -f bfs result.txt
