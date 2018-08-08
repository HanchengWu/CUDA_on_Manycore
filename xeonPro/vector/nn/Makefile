include ../../../common/make.config

ICC = icpc 
 
ICC_FLAGS = -DPROCESSOR -xMIC-AVX512 -qopenmp -O3

#COMMON is where kdtree files are located
COMMON = ../../../common
INC= -I../../../common



nn: kdtree.o main.o 
	$(ICC) $(ICC_FLAGS) -o nn kdtree.o main.o -I$(INCLUDE_PHI) 

main.o: main.cpp
	$(ICC) $(ICC_FLAGS) -c main.cpp  $(INC) -I$(INCLUDE_PHI) 

kdtree.o: $(COMMON)/kdtree.cpp $(COMMON)/kdtree.h
	$(ICC) -c $< -lm $(INC) 

clean:
	rm kdtree.o main.o nn


