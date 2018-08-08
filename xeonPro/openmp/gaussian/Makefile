#include ../../common/make.config

CC = icpc

SRC = gaussian.cpp
EXE = gaussian
CCFLAGS = -g -O3 -qopenmp -D__PURE_INTEL_C99_HEADERS__ 

release: $(SRC)
	$(CC) $(CCFLAGS)  $(SRC) -o $(EXE)   

clean:
	rm gaussian
