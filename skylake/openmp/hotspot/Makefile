# C compiler
CC = icpc
CC_FLAGS = -g -qopenmp -O3

all: hotspot 

hotspot: hotspot_openmp.cpp Makefile 
	$(CC) $(CC_FLAGS) hotspot_openmp.cpp -o hotspot 

clean:
	rm -f hotspot
