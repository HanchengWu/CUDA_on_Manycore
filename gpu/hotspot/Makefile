include ../../common/make.config

CC := nvcc

SRC = hotspot.cu

EXE = hotspot

release: $(SRC)
	$(CC) $(SRC) -o $(EXE)  

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
