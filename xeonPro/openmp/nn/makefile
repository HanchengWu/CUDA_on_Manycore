NVCC=nvcc
CC=g++
#ICC =g++ 
ICC = icpc 

#ICC_FLAGS = -fopenmp 
ICC_FLAGS = -qopenmp -O3


#COMMON is where kdtree files are located
COMMON = ../../Common
INC= -I../../Common

test: kdtree.o main.o
	$(ICC) $(ICC_FLAGS) -o test kdtree.o main.o 

main.o: main.cpp
	$(ICC) $(ICC_FLAGS) -c main.cpp  $(INC)

kdtree.o: $(COMMON)/kdtree.cpp $(COMMON)/kdtree.h
	$(ICC) -c $< -lm $(INC) 

clean:
	rm kdtree.o main.o test


