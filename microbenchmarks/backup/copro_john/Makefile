include ../../common/make.config

CC = icpc 

ICC_FLAGS = -qopenmp -O3

SRC = main.cpp

EXE = main

TPL = ../../common/offload\ library/phi_template.cpp
TPL_H = ../../common/offload\ library/phi_template.h

INCLUDE_PHI = ../../common/offload\ library/

release: $(SRC) phi_template.o
	$(CC) $(ICC_FLAGS) $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI) 

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) 

clean: $(SRC)
	rm -f $(EXE) phi_template.o

