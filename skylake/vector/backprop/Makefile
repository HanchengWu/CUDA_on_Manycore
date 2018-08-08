include ../../../common/make.config

CC := icpc

ICC_FLAGS = -xCORE-AVX512 -qopenmp

SRC = backprop.cpp

EXE = backprop

release: $(SRC) phi_template.o
	$(CC) -DPROCESSOR -DSKYLAKE -D__PURE_INTEL_C99_HEADERS__ $(ICC_FLAGS) $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI) 

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) -DPROCESSOR -DSKYLAKE -D__PURE_INTEL_C99_HEADERS__ $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) 

clean: $(SRC)
	rm -f $(EXE) phi_template.o
