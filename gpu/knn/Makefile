LOCAL_CC = gcc -g -O3 -Wall
CC = nvcc

all : knn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

knn : nn_cuda.cu
	$(CC) -cuda nn_cuda.cu
	$(CC) -o knn nn_cuda.cu

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
