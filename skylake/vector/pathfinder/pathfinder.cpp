#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <string.h>

#define DEBUG 0
#define VERIFY 0

//PHI OFFLOAD STARTS
#include "phi_template.h"

//include all kernel.h here
#include "dynproc_kernel.h"

#define DEVICE 1

//#define BENCH_PRINT

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 7 
int pyramid_height;

int affinity;
float usage;

//#define BENCH_PRINT

void init(int argc, char** argv) {
  if(argc==6){

    affinity = atoi(argv[1]);
    usage = atof(argv[2]);

    rows = atoi(argv[3]);
    cols = atoi(argv[4]);
    pyramid_height=atoi(argv[5]);
  }
  else{
    printf("Usage: affinity usage dynproc row_len col_len pyramid_height\n");
    exit(0);
  }

  data = new int[rows*cols];
  wall = new int*[rows];

  for(int n=0; n<rows; n++)
    wall[n]=data+cols*n;
  result = new int[cols];

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      wall[i][j] = rand() % 10;
    }
  }

//  FILE *file = fopen("output_results_pro.txt", "w");
//  
//  fprintf(file, "wall:\n");
//  for (int i = 0; i < rows; i++) {
//      for (int j = 0; j < cols; j++) {
//  	fprintf(file, "%d ", wall[i][j]);
//      }
//      fprintf(file, "\n");
//  }
//  
//  fclose(file);

//#ifdef BENCH_PRINT
//  for (int i = 0; i < rows; i++) {
//    for (int j = 0; j < cols; j++) {
//      printf("%d ",wall[i][j]) ;
//    }
//    printf("\n") ;
//  }
//#endif

}

void fatal(char *s) {
  fprintf(stderr, "error: %s\n", s);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

/**
 * compute N time steps
 */
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, int pyramid_height, int blockCols, int borderCols) {
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(blockCols);


  FILE *fpout = fopen("pathfinder_skylake_vector_times.txt","a+");
  if(!fpout) {
    printf("Error Saving stats\n");
    exit(1);
  }

  // if file is empty then give it headers
  fseek (fpout, 0, SEEK_END); //move pointer to the end of file
  if ( ftell(fpout) == 0 ) { // if the pos is 0, it is empty
    fprintf(fpout,"%-10s, %-10s, ", "Affinity", "Usage");
    fprintf(fpout,"%-10s, %-10s, %-10s, ", "# rows", "# cols", "height");

    // application specific
    //fprintf(fpout,"rows, cols, height, ");

    fprintf(fpout,"%-20s\n", "time");
  }
  fseek(fpout, 0, SEEK_SET); // reset pointer to start of file

  double start_time = gettime_ms();

  int src = 1, dst = 0;
  for (int t = 0; t < rows-1; t+=pyramid_height) {
    int temp = src;
    src = dst;
    dst = temp;
    dynproc_kernel( affinity, usage, dimGrid, dimBlock, MIN(pyramid_height, rows-t-1), 
          gpuWall, gpuResult[src], gpuResult[dst], cols,rows, t, borderCols);
  }

  double end_time = gettime_ms();
  //printf("Compute time: %lf\n", (end_time - start_time));

  fprintf(fpout,"%-10d, %-10lf, ", affinity, usage);
  fprintf(fpout,"%-10d, %-10d, %-10d, ", rows, cols, pyramid_height);
  //fprintf(fpout,"%d ", no_of_nodes);
  fprintf(fpout,"%-20lf\n", (end_time-start_time));

  fclose(fpout);

  return dst;
}

int main(int argc, char** argv) {
  phiSetDevice(DEVICE);
  run(argc,argv);
  return EXIT_SUCCESS;
}

void run(int argc, char** argv) {
  init(argc, argv);


  /* --------------- pyramid parameters --------------- */
  int borderCols = (pyramid_height)*HALO;
  int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
  int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

  printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
  pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

  int *gpuWall, *gpuResult[2];
  int size = rows*cols;

//  phiMalloc(&gpuResult[0], cols);
//  phiMalloc(&gpuResult[1], cols);
//  phiMemcpy(gpuResult[0], data, cols, CpuToPhi);
//  phiMalloc(&gpuWall, (size-cols));
//  phiMemcpy(gpuWall, data+cols, (size-cols), CpuToPhi);

  gpuResult[0] = (int *)malloc(cols*sizeof(int));
  gpuResult[1] = (int *)malloc(cols*sizeof(int));

  memcpy(gpuResult[0], data, cols*sizeof(int));

  gpuWall = (int*)malloc((size-cols)*sizeof(int));

  memcpy(gpuWall, (data+cols), (size-cols)*sizeof(int));

  int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
                            pyramid_height, blockCols, borderCols);

//  phiMemcpy(result, gpuResult[final_ret], cols, PhiToCpu);
  memcpy(result, gpuResult[final_ret],cols*sizeof(int));


//#ifdef BENCH_PRINT
//  for (int i = 0; i < cols; i++)
//    printf("%d ",data[i]) ;
//  printf("\n") ;
//  for (int i = 0; i < cols; i++)
//    printf("%d ",result[i]) ;
//  printf("\n") ;
//#endif

//        FILE *file = fopen("output_results_pro.txt", "w");
//
//        fprintf(file, "data:\n");
//        for (int i = 0; i < cols; i++)
//            fprintf(file, "%d ", data[i]);
//        fprintf(file, "\n");
//
//        fprintf(file, "result:\n");
//        for (int i = 0; i < cols; i++)
//            fprintf(file, "%d ", result[i]);
//        fprintf(file, "\n");
//
//        fclose(file);


  //Store the result into a file
  FILE *fpo = fopen("result.txt","w");
  for (int i = 0; i < cols; i++)
    fprintf(fpo, "%d ",result[i]) ;
  fprintf(fpo,"\n") ;
  fclose(fpo);
  printf("Result stored in result.txt\n");

  free(gpuWall);
  free(gpuResult[0]);
  free(gpuResult[1]);

  delete [] data;
  delete [] wall;
  delete [] result;
}

