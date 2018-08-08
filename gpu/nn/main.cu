#include "kdtree.h"
#include "cuda.h"
#include <cfloat>
#include <sys/time.h>
#include "utility.h"

#define gridsize 208
#define blocksize 192

__device__ unsigned DIM;
__device__ unsigned OBN;

int verify=0;
int QS=64;

int dim;
int objnum;
 
inline double gettime_ms() { 
        struct timeval t; 
        gettimeofday(&t,NULL); 
        return (t.tv_sec+t.tv_usec*1e-6)*1000; 
} 

__device__ int readvalue(int dim, int obj, int *in_o){
        return in_o[dim*OBN+obj];
}


__device__ int nn_rec(int curr, int depth,int *g_guess, float *bestDist, int query, 	int* vtxarr, int*edgearr, int*in_o);



__global__ void  nn(int *g_guess,  float *bestDist, int *query,int q_len, int root,  int *vtxarr, int *edgearr, int *in_o){
        //see if root set, otherwise return
        //recursively call nn_rec
	int TID=blockDim.x*blockIdx.x+threadIdx.x;	
	int stride=gridDim.x*blockDim.x;

	for(int tid=TID; tid< q_len ;tid+=stride){ //this line replaces the next line to make the kernel launch configuration configurable
//	if (tid < q_len) {
//		printf("\nI am thread %d.OBJNUM: %d, DIM: %d. I take query:%d. \n",tid,OBN,DIM, query[tid]);
        	g_guess[tid]=UNDEFINED;
       		bestDist[tid]=FLT_MAX;
		nn_rec(root,0,g_guess+tid,bestDist+tid,query[tid],	vtxarr,edgearr,in_o);
//		printf("\nThe nn of query node %d is node %d.\n", query[tid], g_guess[tid]);
	}
}


__device__ int nn_rec(int curr, int depth,int *g_guess, float *bestDist, int query, 	int* vtxarr, int*edgearr, int*in_o){
        if( UNDEFINED==curr )
                return 0;
        int e_p=vtxarr[curr];
        int left=edgearr[e_p];
        int right=edgearr[e_p+1];
	//update best distance
        unsigned dis_sq=.0;

        for(int i=0;i<DIM;++i){
                float tmp=readvalue(i,curr,in_o)-readvalue(i,query,in_o);
                dis_sq+=tmp*tmp;
        }

        float dis=sqrtf(dis_sq);

        if (dis<*bestDist && dis !=.0){                
		*bestDist = dis;
                *g_guess = curr;
        }
        int dim=depth%DIM;

        //if ai<curri
	int diff=readvalue(dim,curr,in_o)-readvalue(dim,query,in_o);
        if (diff>0) {
                nn_rec(left,depth+1, g_guess, bestDist, query, 	vtxarr, edgearr, in_o);
                if (diff < *bestDist)
                        nn_rec(right,depth+1, g_guess, bestDist, query, 	vtxarr, edgearr, in_o);
        }else{ 
                nn_rec(right,depth+1, g_guess, bestDist, query, 	vtxarr, edgearr, in_o);
                if ( (-1)*diff < *bestDist )
                        nn_rec(left,depth+1, g_guess, bestDist, query, 	vtxarr, edgearr, in_o);
        }
	return 0;
}


int main(int argc, char *argv[]){

	if (5!= argc) {
		printf("Arguments is less then enough!\n");
		exit(0);
}
	double time;
	dim=7;
	int objnum=atoi(argv[1]);
	QS=objnum;
	int balanced=atoi(argv[2]);
	int random=atoi(argv[3]);
	int sortin=atoi(argv[4]);
	printf("Dim:%d. OBJNUM:%d. QueryNum:%d. Random:%d. Balanced tree:%d, Sorted:%d.\n",dim,objnum,QS,random,balanced, sortin);
	kdtree mytree;
	mytree.set_tree(dim,objnum);
	mytree.setbalance(balanced);

	cudaSetDevice(1);

	cudaMemcpyToSymbol(DIM,&dim,sizeof(int));
	cudaMemcpyToSymbol(OBN,&objnum,sizeof(int));

	//read stack limit in bytes per thread on device
	size_t slimit;
	cudaDeviceGetLimit(&slimit,cudaLimitStackSize);
	printf("\nDefault per thread stack size:%d Kb.\n",slimit/1024);

	//set new stack limit in bytes/thread on device
    slimit=1024*4; 
	cudaDeviceSetLimit(cudaLimitStackSize,slimit);

	cudaDeviceGetLimit(&slimit,cudaLimitStackSize);
	printf("\nUpdated per thread stack size:%d Kb.\n",slimit/1024);

	if(random==0){
		mytree.read_cov("../Input/covtype.data");
	}else{
		mytree.gnrt_random();	
	}
	
// 	mytree.build_tree_from_txt();
 	mytree.build_tree();
	printf("The root node is:%d. Max depth is %d.\n", mytree.get_root(), mytree.max_depth);

	//allocate and copy tree into GPU
	int *vtxarr_d;
	int *edgearr_d;
	int *in_d;

	cudaMalloc(&vtxarr_d,objnum*sizeof(int));	
	cudaMalloc(&edgearr_d,2*objnum*sizeof(int));	
	cudaMalloc(&in_d,dim*objnum*sizeof(int));

	cudaMemcpy(vtxarr_d,mytree.return_vtxarr(),objnum*sizeof(int),cudaMemcpyHostToDevice);	
	cudaMemcpy(edgearr_d,mytree.return_edgearr(),2*objnum*sizeof(int),cudaMemcpyHostToDevice);	
	cudaMemcpy(in_d,mytree.return_in_o(),dim*objnum*sizeof(int),cudaMemcpyHostToDevice);	


	int *query_d;
	int *guess_d;
	float *bestdis_d;
	cudaMalloc(&query_d,QS*sizeof(int));
	cudaMalloc(&guess_d,objnum*sizeof(int));
	cudaMalloc(&bestdis_d,objnum*sizeof(float));	

	int *query=(int *)malloc(QS*sizeof(int));
	int *guess=(int *)malloc(QS*sizeof(int));
	float *bestdis=(float *)malloc(QS*sizeof(float));

    //printLastCudaError('A');


	for(int i=0;i<QS;++i)
		query[i]=i;
	
	printf("\nQueries initialized!\n",QS);
	if(sortin==1){
		printf("queries are sorted!");
		mytree.sort_input(query,QS);
	}

	printf("\nBegin querying on GPU\n");
	cudaMemcpy(query_d,query,QS*sizeof(int),cudaMemcpyHostToDevice);

        FILE *fresult = fopen("result_gpu.txt","a+");

	time=gettime_ms();

	nn<<<gridsize,blocksize>>>(guess_d,bestdis_d, query_d, QS, mytree.get_root(),  vtxarr_d, edgearr_d, in_d);
        
	printLastCudaError('B');
	
	cudaDeviceSynchronize();

        double e_time=gettime_ms()-time;

        fprintf(fresult,"%-8d, %-20lf\n",objnum, e_time);

	fclose(fresult);

 	printf("\nQuerying on GPU returns with time = %.2f ms.\n", gettime_ms()-time);

	cudaMemcpy(guess,guess_d,QS*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(bestdis,bestdis_d,QS*sizeof(int),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();


        //printLastCudaError('C');
        if(verify){
                int *guess_v=(int *)malloc(QS*sizeof(int));
                float *bestdis_v=(float *)malloc(QS*sizeof(float));
                printf("\nverification begins!\n");
                time = gettime_ms();
                mytree.nn(query,guess_v,bestdis_v,QS);
                printf("\nQuerying on CPU returns with time = %.2f ms.\n",gettime_ms()-time);
                printf("\n");

                for (int i=0;i<QS;++i){
                   float tem=fabs(bestdis[i]-bestdis_v[i]);
                   if(tem>0.00001){
                        printf("\nVerification failed at query %d. GPU bestdis %.5f. CPU bestdis %.5f. The diff is %.5f.\n",i,bestdis[i],bestdis_v[i],tem);
			return 1;
                   }
                }
                free(guess_v);
                free(bestdis_v);

                printf("\nVefication passed\n");
        }


	cudaFree(vtxarr_d);
	cudaFree(edgearr_d);
	cudaFree(in_d);
	cudaFree(guess_d);
	cudaFree(bestdis_d);

	free(query);
	free(guess);
	free(bestdis);
	return 0;
}



















