include ../../../common/make.config

ICC = icpc 
 
ICC_FLAGS = -qopenmp -O3

#COMMON is where kdtree files are located
COMMON = ../../../common
INC= -I../../../common



nn: kdtree.o main.o phi_template.o
	$(ICC) $(ICC_FLAGS) -o nn kdtree.o main.o phi_template.o -I$(INCLUDE_PHI) 

main.o: main.cpp
	$(ICC) $(ICC_FLAGS) -c main.cpp  $(INC) -I$(INCLUDE_PHI) 

kdtree.o: $(COMMON)/kdtree.cpp $(COMMON)/kdtree.h
	$(ICC) -c $< -lm $(INC) 

phi_template.o: $(TPL) $(TPL_H) 
	$(ICC) $(ICC_FLAGS) -g -c $(TPL) -I$(INCLUDE_PHI) -I$(INCLUDE_PHI) 
clean:
	rm kdtree.o main.o nn


