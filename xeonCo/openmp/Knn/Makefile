LOCAL_CC = icpc -g -O3 
CC = icpc
CFLAGS = -lm -qopenmp -Wall


all : nn

clean :
	rm -rf *.o nn

nn : nn_openmp.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) 

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

