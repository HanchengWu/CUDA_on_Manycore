include ../../../common/make.config

CC := icpc

ICC_FLAGS = -DSKYLAKE -xCORE-AVX512 -qopenmp -O3

SRC = pathfinder.cpp

EXE = pathfinder

release: $(SRC) phi_template.o
	$(CC) -DPROCESSOR $(ICC_FLAGS) $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI) 

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) -DPROCESSOR $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) 

clean: $(SRC)
	rm -f $(EXE) phi_template.o

