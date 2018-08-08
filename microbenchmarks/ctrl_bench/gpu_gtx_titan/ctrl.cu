#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

inline double gettime_ms() {
        struct timeval t;
        gettimeofday(&t,NULL);
        return (t.tv_sec+t.tv_usec*1e-6)*1000;
}

__device__ int cost_func(int costX){
        int tem=0;
	for(int i=0;i<costX;++i){
		tem++;
	}
	return tem;
}

__global__ void ctrl_kernel(int D, int* res_arr, int costX) { 

  int result=0; 

  for(int i=0; i<D; ++i){ //D’s maximum value is 4
     int tmp=1;
     for (int j=0;j<4-i;++j) tmp=tmp*2;
     int t=16-tmp;
     if(threadIdx.x<16-t){
       result+=cost_func(costX);
     }else if (threadIdx.x>=16+t){
       result+=cost_func(costX);
     } 
  }
  res_arr[blockIdx.x*blockDim.x+threadIdx.x]=result;
}

int main(int argc, char **argv){

//input: Utilization U, Affinity A, BlockNum B, ThreadsNum T, Cost X, Divergen D
int U, A, B, T, X, D;
if (argc!=7) {
	printf("\nInput arguments wrong!\n input: Utilization U, Affinity A, BlockNum B, ThreadsNum T, Cost X, Divergence D  \n ");
	return 0;
}
U=atof(argv[1]);
A=atoi(argv[2]);
B=atoi(argv[3]);
T=atoi(argv[4]);
X=atoi(argv[5]);
D=atoi(argv[6]);

if(D>4||D<0) {
	printf("\nDivergence D has to be between 0~4\n");
	return 0;
}
cudaSetDevice(0);

int *res_arr_h=(int *)malloc(B*T*sizeof(int));
int *res_arr_d;
cudaMalloc(&res_arr_d,B*T*sizeof(int));

printf("\nkernel starts\n");
double ktime=gettime_ms();
ctrl_kernel<<<B,T>>>(D,res_arr_d,X);
cudaDeviceSynchronize();
ktime=gettime_ms()-ktime;

FILE* fp=fopen("result.txt","a+");
fprintf(fp,"%f ",ktime);
fclose(fp);

printf("\nKernel_Finish_in %f ms\n",ktime);
cudaMemcpy(res_arr_h,res_arr_d,B*T*sizeof(int),cudaMemcpyDeviceToHost);
printf("results:\n");
for(int i=0;i<32;++i){
        printf("%d ",res_arr_h[i]);
}
printf("\n");

free(res_arr_h);
cudaFree(res_arr_d);
return 0;
}
