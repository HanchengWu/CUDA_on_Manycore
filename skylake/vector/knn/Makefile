include ../../../common/make.config

ICC_FLAGS = -DPROCESSOR -DSKYLAKE -xCORE-AVX512  -qopenmp -O3

CC := icpc

all : knn 

clean :
	rm -rf *.o knn 

knn : nn_cuda.cpp phi_template.o
	$(CC) $(ICC_FLAGS) -o knn nn_cuda.cpp phi_template.o -I$(INCLUDE_PHI) 

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) 

