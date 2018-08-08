NVCC=nvcc
CC=g++
#ICC =g++ 
ICC = icpc 

#ICC_FLAGS = -fopenmp 
ICC_FLAGS = -qopenmp -O3


#COMMON is where kdtree files are located
COMMON = ../common
INC= -I../common

test: main.o
	$(ICC) $(ICC_FLAGS) -o test main.o 

main.o: main.cpp
	$(ICC) $(ICC_FLAGS) -c main.cpp  $(INC)

clean:
	rm main.o test


