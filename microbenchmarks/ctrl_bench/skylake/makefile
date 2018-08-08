CC = icpc 

ICC_FLAGS = -DPROCESSOR -DSKYLAKE -xCORE-AVX512 -qopenmp -O3

SRC = ctrl.cpp

EXE = ctrl

release: $(SRC) 
	$(CC) $(ICC_FLAGS) $(SRC)  -o $(EXE) -I$(INCLUDE_PHI) 

clean: 
	rm -f $(EXE) 

