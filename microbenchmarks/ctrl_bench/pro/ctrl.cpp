#include "phi_template.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DEBUG 0
#include "ctrl_kernel.h"

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


	printf("\nInput: Utilization U:%d, Affinity A:%d, BlockNum B:%d, ThreadsNum T:%d, Cost X:%d, Divergence D:%d  \n ", U,A,B,T,X,D);

//phiSetDevice(1);

int *res_arr_h=(int *)malloc(B*T*sizeof(int));
int *res_arr_d;
//phiMalloc(&res_arr_d,B*T);

printf("\nkernel starts\n");
double ktime=gettime_ms();
ctrl_kernel(1,1.0,B,T,D,res_arr_h,X);
//cudaDeviceSynchronize();
ktime=gettime_ms()-ktime;
printf("\nKernel_Finish_in %f ms\n",ktime);

//phiMemcpy(res_arr_h,res_arr_d,B*T,PhiToCpu);
printf("results:\n");
for(int i=0;i<32;++i){
        printf("%d ",res_arr_h[i]);
}
printf("\n");

free(res_arr_h);
//phiFree(res_arr_d);
return 0;
}
