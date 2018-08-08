include ../../../common/make.config

CC = icpc 

ICC_FLAGS = -DPROCESSOR -xMIC-AVX512 -qopenmp -O3

SRC = sync.cpp

EXE = sync

all: $(SRC) phi_template.o
	$(CC) $(ICC_FLAGS) -DREAD -DSYNC -DWRITE  $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI) 


sync_only: $(SRC) phi_template.o
	$(CC) $(ICC_FLAGS) -DSYNC $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI) 


mem_only: $(SRC) phi_template.o
	$(CC) $(ICC_FLAGS) -DREAD -DWRITE $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI) 

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) 

clean: 
	rm -f $(EXE) phi_template.o

