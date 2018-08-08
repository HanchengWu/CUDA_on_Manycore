include ../../../common/make.config

ICC_FLAGS = -qopenmp -O3

CC := icpc

SRC = gaussian.cpp
EXE = gaussian

release: $(SRC) phi_template.o
	$(CC) $(ICC_FLAGS) $(KERNEL_DIM) $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI)

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI)

clean:
	rm gaussian
