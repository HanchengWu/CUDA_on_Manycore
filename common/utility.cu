#include <utility.h>

void printLastCudaError(char c){
	printf("\nAt %c: %s\n", c, cudaGetErrorString( cudaGetLastError() ));
}
