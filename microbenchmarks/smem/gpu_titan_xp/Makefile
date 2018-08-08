NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -Xcompiler -fopenmp -g -w -O3 -I.. 
LINK_OPT = -L/usr/local/cuda/lib64 -lcuda -lcudart

all:	mem

clean:
	-rm mem

mem : mem.cu 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} mem.cu -o mem

