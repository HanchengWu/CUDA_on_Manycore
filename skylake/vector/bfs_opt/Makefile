include ../../../common/make.config

CC = icpc 

ICC_FLAGS = -DSKYLAKE -D__PURE_INTEL_C99_HEADERS__  -xCORE-AVX512 -qopenmp -O3

SRC = bfs.cpp

EXE = bfs

release: $(SRC) phi_template.o
	$(CC) -DPROCESSOR $(ICC_FLAGS) $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI) 

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) -DPROCESSOR $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) 

clean: $(SRC)
	rm -f $(EXE) phi_template.o result.txt

