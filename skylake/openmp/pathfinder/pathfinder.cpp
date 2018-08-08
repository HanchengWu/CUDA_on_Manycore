#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <omp.h>
#include "timer.h"
#include <sys/time.h>

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

//#define BENCH_PRINT

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9

inline double gettime_ms() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return (t.tv_sec+t.tv_usec*1e-6)*1000;
}

void
init(int argc, char** argv)
{
	if(argc==4){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
	}else{
                printf("Usage: pathfiner width num_of_steps\n");
                exit(0);
        }
	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
    for (int j = 0; j < cols; j++)
        result[j] = wall[0][j];
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int main(int argc, char** argv)
{
    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);
    int num_omp_threads=atoi(argv[3]);

    unsigned long long cycles;

    FILE *fpout = fopen("pathfinder_skylake_omp_times.txt","a+");
    if(!fpout) {
      printf("Error Saving stats\n");
      exit(1);
    }

    // if file is empty then give it headers
    fseek (fpout, 0, SEEK_END); //move pointer to the end of file
    if ( ftell(fpout) == 0 ) { // if the pos is 0, it is empty
      fprintf(fpout,"%-10s, ", "#threads");
      fprintf(fpout,"%-10s, %-10s, ", "# rows", "# cols");

      fprintf(fpout,"%-20s\n", "time");
    }
    fseek(fpout, 0, SEEK_SET); // reset pointer to start of file

    double start_time = gettime_ms();

    int *src, *dst, *temp;
    int min;

    dst = result;
    src = new int[cols];

    pin_stats_reset();
    omp_set_num_threads(num_omp_threads);
    for (int t = 0; t < rows-1; t++) {
        temp = src;
        src = dst;
        dst = temp;
        #pragma omp parallel for private(min)
        for(int n = 0; n < cols; n++){
          min = src[n];
          if (n > 0)
            min = MIN(min, src[n-1]);
          if (n < cols-1)
            min = MIN(min, src[n+1]);
          dst[n] = wall[t+1][n]+min;
        }
    }

    double end_time = gettime_ms();
    //printf("Compute time: %lf\n", (end_time - start_time));

    fprintf(fpout,"%-10d, ", num_omp_threads);
    fprintf(fpout,"%-10d, %-10d, ", rows, cols);
    //fprintf(fpout,"%d ", no_of_nodes);
    fprintf(fpout,"%-20lf\n", (end_time-start_time));
    fclose(fpout);

    pin_stats_pause(cycles);
    pin_stats_dump(cycles);

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (int i = 0; i < cols; i++)
            printf("%d ",dst[i]) ;
    printf("\n") ;
#endif

  //Store the result into a file
//  FILE *fpo = fopen("result.txt","w");
//  for (int i = 0; i < cols; i++)
//    fprintf(fpo, "%d ",result[i]) ;
//  fprintf(fpo,"\n") ;
//  fclose(fpo);
//  printf("Result stored in result.txt\n");


    delete [] data;
    delete [] wall;
    delete [] dst;
    delete [] src;
}

