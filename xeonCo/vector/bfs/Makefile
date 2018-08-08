include ../../../common/make.config

CC = icpc 

ICC_FLAGS = -qopenmp -O3

SRC = bfs.cpp

EXE = bfs

release: $(SRC) phi_template.o
	$(CC) $(ICC_FLAGS) $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI) 

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) 

clean: $(SRC)
	rm -f $(EXE) phi_template.o

