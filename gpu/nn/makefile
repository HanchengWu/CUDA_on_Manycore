NVCC=nvcc
CC=g++
NVCCFLAG= -g -O3

#COMMON is where kdtree1 files are located
COMMON = ../../common
INC= -I../../common

test: kdtree1.o main.o utility.o
	$(NVCC) $(NVCCFLAG) -o test kdtree1.o main.o utility.o

main.o: main.cu
	$(NVCC) $(NVCCFLAG) -c main.cu $(INC)

kdtree1.o: $(COMMON)/kdtree1.cpp $(COMMON)/kdtree1.h 
	$(CC) -c $< -lm $(INC) 

utility.o: $(COMMON)/utility.cu $(COMMON)/utility.h
	$(NVCC) -c $< -lm $(INC) 

clean:
	rm *.o test

