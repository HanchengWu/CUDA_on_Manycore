include ../../../common/make.config

ICC_FLAGS = -qopenmp -O3

LOCAL_CC = icpc -g -O3 -Wall
CC := icpc

all : knn hurricane_gen

clean :
	rm -rf *.o knn hurricane_gen

knn : nn_cuda.cpp phi_template.o
#	$(CC) -cuda nn_cuda.cu
	$(CC) $(ICC_FLAGS) -o knn nn_cuda.cpp phi_template.o -I$(INCLUDE_PHI) 

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) 

#data :
#	mkdir data
#	./gen_dataset.sh
