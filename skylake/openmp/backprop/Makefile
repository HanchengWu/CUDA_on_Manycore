# C compiler
CC =gcc
CC_FLAGS = -g -D__PURE_INTEL_C99_HEADERS__ -fopenmp  -O2

backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -o backprop -lm

clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
