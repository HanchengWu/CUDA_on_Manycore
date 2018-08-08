include ../../../common/make.config

CC := icpc

ICC_FLAGS = -qopenmp -O3

SRC = hotspot.cpp

EXE = hotspot

KERNEL_DIM = -DRD_WG_SIZE_0=16

release: $(SRC) phi_template.o
	$(CC) $(KERNEL_DIM) $(ICC_FLAGS) $(SRC) phi_template.o -o $(EXE) -I$(INCLUDE_PHI) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) $(ICC_FLAGS) -g $(SRC) -o $(EXE) -I$(INCLUDE_PHI) 

phi_template.o: $(TPL) $(TPL_H) 
	$(CC) $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt output.out phi_template.o
