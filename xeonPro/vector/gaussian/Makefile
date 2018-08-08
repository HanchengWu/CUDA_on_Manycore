include ../../../common/make.config

ICC_FLAGS = -DPROCESSOR -xMIC-AVX512  -qopenmp -O3

CC := icpc

SRC = gaussian.cpp
EXE = gaussian

release: $(SRC) 
	$(CC) $(ICC_FLAGS) $(SRC) -o $(EXE) -I$(INCLUDE_PHI)

clean:
	rm gaussian
