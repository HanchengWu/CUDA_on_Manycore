NVCC = /usr/local/cuda/bin/nvcc

NVCCFLAGS = -Xcompiler -fopenmp -g -w -O3 -I.. 

LINK_OPT = -L/usr/local/cuda/lib64 -lcuda -lcudart

all:	sync

clean:
	-rm sync

sync : sync.cu 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} -DREAD -DSYNC -DWRITE sync.cu -o sync

sync_only : sync.cu 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT}  -DSYNC sync.cu -o sync

mem_only : sync.cu 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} -DREAD -DWRITE sync.cu -o sync
