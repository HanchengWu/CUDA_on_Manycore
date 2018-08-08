#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

//PHI Headers
#define DEBUG 0
#define VERIFY 0
#include "phi_template.h"
//Ends

#define MAX_THREADS_PER_BLOCK 16 

//include all kernel.h here
#include "Kernel.h"

int affinity;
float usage;

int main( int argc, char** argv) {

  char *input_f;
	if(argc!=3){
    fprintf(stderr,"Usage: %s <affinity> <usage>\n", argv[0]);
    exit(0);
	}

  // initialize random seed: 
  srand (time(NULL));

  //generate secret number between 1 and 10:
  int x = rand() % 10 + 1;

  affinity = atoi(argv[1]);
  usage = atof(argv[2]);
	
	int source = 0;

	phiSetDevice(1);

  int num_of_threads_total = 16;

	int num_of_blocks = 1;
	int num_of_threads_per_block = num_of_threads_total;

	if(num_of_threads_total>MAX_THREADS_PER_BLOCK) {
		num_of_blocks = (int)ceil(num_of_threads_total/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

  int n = 500000000;
  int *h_array = (int *) malloc(sizeof(int)*n);

	for( unsigned int i = 0; i < num_of_threads_total; i++) {
		h_array[i]=1;
	}

  int *d_array;
  phiMalloc( &d_array, n );
  phiMemcpy( d_array, h_array, n, CpuToPhi );

	printf("Copied Everything to Xeon Phi memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

  double start_time = gettime_ms();

  int sum = 0;
  Kernel( affinity, usage, grid, threads, d_array, sum);
  printf("sum: %d\n", sum);

  //phiMemcpy( &stop, d_over, 1, PhiToCpu) ;
	double end_time = gettime_ms();
  printf("Compute time: %-20lf\n", (end_time - start_time));

  //fprintf(fpout,"%-10d, %-10lf, ", affinity, usage);
  //fprintf(fpout,"%d ", num_of_threads_total);
  //fprintf(fpout,"%-20lf\n", (end_time-start_time));

	// cleanup memory
	free( h_array );

	phiFree(d_array);
}

