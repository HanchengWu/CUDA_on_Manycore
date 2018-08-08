#include "phi_template.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DEBUG 0
#include "mem_kernel.h"


int main(int argc, char **argv){

//input: Size N, Utilization U, Affinity A, BlockNum B, ThreadsNum T, Randomizer X
int N, U, A, B, T, X;
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

printf("\ninput: Size N:%d, Utilization U:%d, Affinity A:%d, BlockNum B:%d, ThreadsNum T:%d, Random X:%d  \n ",N,U,A,B,T,X);

if(N%32!=0) {
	printf("\nArray size N has to be multiple of 32\n");
	return 0;
}

//phiSetDevice(1);

int *array_h=(int *)malloc(N*sizeof(int));
for (int i=0;i<N;++i) array_h[i]=1;
int *array_d;
//phiMalloc(&array_d,N);
//phiMemcpy(array_d,array_h,N,CpuToPhi);

int *rdom_arr_h=(int *)malloc(N*sizeof(int));
for (int i=0;i<N;++i) rdom_arr_h[i]= rand() % X; //generate random number in range [0, X)
int *rdom_arr_d;
//phiMalloc(&rdom_arr_d,N);
//phiMemcpy(rdom_arr_d,rdom_arr_h,N,CpuToPhi);

int *result_h=(int *)malloc(B*T*sizeof(int));
int *result_d;
//phiMalloc(&result_d,B*T);

double ktime=gettime_ms();
mem_kernel(1, 1.0, B, T, array_h,N,rdom_arr_h,result_h);
//cudaDeviceSynchronize();
ktime=gettime_ms()-ktime;
printf("\nKernel_Time %f ms\n",ktime);

//phiMemcpy(result_h,result_d,B*T,PhiToCpu);
//cudaDeviceSynchronize();
printf("results:\n");
for(int i=0;i<10;++i){
	printf("%d ",result_h[i]);
}
printf("\n");
free(array_h);
free(rdom_arr_h);
free(result_h);
//phiFree(array_d);
//phiFree(rdom_arr_d);
//phiFree(result_d);
return 0;
}
