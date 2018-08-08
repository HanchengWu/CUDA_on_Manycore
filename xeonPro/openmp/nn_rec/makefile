ICC = icpc 

ICC_FLAGS = -D__PURE_INTEL_C99_HEADERS__ -qopenmp -O3


#COMMON is where kdtree files are located
COMMON = ../../../common
INC= -I../../../common

test: kdtree.o main.o
	$(ICC) $(ICC_FLAGS) -o test kdtree.o main.o 

main.o: main.cpp
	$(ICC) $(ICC_FLAGS) -c main.cpp  $(INC)

kdtree.o: $(COMMON)/kdtree.cpp $(COMMON)/kdtree.h
	$(ICC) $(ICC_FLAGS) -c $< -lm $(INC) 

clean:
	rm -rf kdtree.o main.o test


