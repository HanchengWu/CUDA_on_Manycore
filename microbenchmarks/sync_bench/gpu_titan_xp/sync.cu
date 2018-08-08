#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

inline double gettime_ms() {
        struct timeval t;
        gettimeofday(&t,NULL);
        return (t.tv_sec+t.tv_usec*1e-6)*1000;
}

__global__ void mem_kernel(int *arr, int N, int *rdom_arr, int *result) { 
  int tem=0;
  int random;
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  int totalthreads=blockDim.x*gridDim.x;

  int j=tid;
  for(int i=0;i<100000;++i){
/////////////////////////////
#ifdef READ
      int idx1 = rdom_arr[j%N];

      int idx2 = (j+idx1)%N;

      tem += arr[idx2];
#endif 
////////////////////////////
#ifdef SYNC
      __syncthreads();
#endif
////////////////////////////
#ifdef WRITE
      result[idx2]=tem;    
#endif
////////////////////////////
      j+=totalthreads;
  } 
  //write result
}

int main(int argc, char **argv){

//input: Size N, Utilization U, Affinity A, BlockNum B, ThreadsNum T, Randomizer X
int N, A, B, T, X;
float U;
if (argc!=7) {
	printf("\nInput arguments wrong!\n input: Size N, Utilization U, Affinity A, BlockNum B, ThreadsNum T, Random X  \n ");
	return 0;
}
N=atoi(argv[1]);
U=atof(argv[2]);
A=atoi(argv[3]);
B=atoi(argv[4]);
T=atoi(argv[5]);
X=atoi(argv[6]);

if(N%32!=0) {
	printf("\nArray size N has to be multiple of 32\n");
	return 0;

}

        printf("\nInput: Size N:%d, Utilization U:%f, Affinity A:%d, BlockNum B:%d, ThreadsNum T:%d, Random X:%d  \n ",N, U, A, B, T, X);
 


cudaSetDevice(0);

int *array_h=(int *)malloc(N*sizeof(int));
for (int i=0;i<N;++i) array_h[i]=1;
int *array_d;
cudaMalloc(&array_d,N*sizeof(int));
cudaMemcpy(array_d,array_h,N*sizeof(int),cudaMemcpyHostToDevice);

int *rdom_arr_h=(int *)malloc(N*sizeof(int));
for (int i=0;i<N;++i) rdom_arr_h[i]= rand() % X; //generate random number in range [0, X)
int *rdom_arr_d;
cudaMalloc(&rdom_arr_d,N*sizeof(int));
cudaMemcpy(rdom_arr_d,rdom_arr_h,N*sizeof(int),cudaMemcpyHostToDevice);

int *result_h=(int *)malloc(N*sizeof(int));
int *result_d;
cudaMalloc(&result_d,N*sizeof(int));

double ktime=gettime_ms();
mem_kernel<<<B,T>>>(array_d,N,rdom_arr_d,result_d);
cudaDeviceSynchronize();
ktime=gettime_ms()-ktime;
printf("\nKernel_Time %f ms\n",ktime);

FILE *fp=fopen("result.txt","a+");
fprintf(fp,"%f ",ktime);
fclose(fp);

cudaMemcpy(result_h,result_d,B*T*sizeof(int),cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
printf("results:\n");
for(int i=0;i<10;++i){
	printf("%d ",result_h[i]);
}
printf("\n");
free(array_h);
free(rdom_arr_h);
free(result_h);
cudaFree(array_d);
cudaFree(rdom_arr_d);
cudaFree(result_d);
return 0;
}
