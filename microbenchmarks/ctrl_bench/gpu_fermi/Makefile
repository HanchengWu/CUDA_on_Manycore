NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -Xcompiler -fopenmp -g -w -O3 -I.. 
LINK_OPT = -L/usr/local/cuda/lib64 -lcuda -lcudart

all:	ctrl

clean:
	-rm ctrl

ctrl : ctrl.cu 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} ctrl.cu -o ctrl

