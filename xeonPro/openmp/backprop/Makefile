# C compiler
CC =icpc
CC_FLAGS = -g -D__PURE_INTEL_C99_HEADERS__ -fopenmp  -O3

backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -o backprop -lm

clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
