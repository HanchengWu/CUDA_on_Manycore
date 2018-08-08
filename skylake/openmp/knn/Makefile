LOCAL_CC = gcc -g -O3 -Wall
CC = icpc 
CFLAGS = -lm -qopenmp -Wall -g -O3 -D__PURE_INTEL_C99_HEADERS__


all : nn

clean :
	rm -rf *.o nn

nn : nn_openmp.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) 

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

