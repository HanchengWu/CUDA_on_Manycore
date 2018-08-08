//Replace "mem_kernel" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __mem_kernel__
#define __mem_kernel__

//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel mem_kernel
    //int para;
    int *arr;
    int N;
    int *rdom_arr;
    int *result;

} PR_mem_kernel;

//Defines the object passed to each pthread instance for mem_kernel
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    int *sm;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_mem_kernel *kp;//define pointer to PR_mem_kernel

} P_PR_mem_kernel;

/*MOD_AUTO*/
void *mem_kernel_KERNEL(void *arg);

void *mem_kernel_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_mem_kernel *p=(P_PR_mem_kernel *)arg;    //restore pthead data pointer
    PR_mem_kernel *kp=p->kp;   //restore mem_kernel config and para pointer
    
    /*MOD_MAN*///restore kernel mem_kernel's parameters
    int *arr = kp->arr;
    int N = kp->N;
    int *rdom_arr = kp->rdom_arr;
    int *result = kp->result;
    int *sm=p->sm;
    #include "zpart3.h"    //recover kernel configuration

    /*MOD_MAN*/ //Activate corresponding threadIdx, all vector-lane variant variables
    vint threadIdxx= _mm512_mask_rem_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x)) ;
    vint threadIdxy= _mm512_mask_rem_epi32(vseti(0),init_mask,_mm512_div_epi32(_threadIdx, vseti(blockDim.x)), vseti(blockDim.y)) ;
    vint threadIdxz= _mm512_mask_div_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x*blockDim.y)) ;

    for(unsigned _bidx=p->blockgroupid; _bidx<numBlocksPerGrid; _bidx += kp->numActiveBlocks){
        /*MOD_MAN*/ //Activate corresponding blockIdx, all vector-lane invariant variables
        Dim3 blockIdx;
        blockIdx.x = _bidx%gridDim.x; 
        blockIdx.y = (_bidx/gridDim.x)%gridDim.y;
        blockIdx.z = _bidx/(gridDim.x*gridDim.y);
/*MOD_MAN*///KERNEL STARTS//
	    vint tem=vseti(0);
	    vint random;
        vint tid= _mm512_mask_add_epi32(vseti(0),init_mask,threadIdxx,vseti(blockIdx.x*blockDim.x));
	    int totalthreads=blockDim.x*gridDim.x;
	vint sum=vseti(0);
 	// sm[threadIdx.x]=arr[tid];
	vint tmp=_mm512_mask_i32gather_epi32(vseti(0),init_mask,tid,arr,sizeof(int));
	_mm512_mask_i32scatter_epi32(sm, init_mask, threadIdxx,tmp,sizeof(int));
    for (int i=0;i<10;++i){

		vint j=threadIdxx;
		vmask mk=_mm512_mask_cmp_epi32_mask(init_mask,j,vseti(N),_MM_CMPINT_LT);
		while(mk!=0){
		random = _mm512_mask_i32gather_epi32(vseti(0),mk,j,rdom_arr,sizeof(int));
		vint idx= _mm512_mask_rem_epi32(vseti(0),mk,  _mm512_mask_add_epi32(vseti(0),mk,j,random),vseti(blockDim.x) );

		tem = _mm512_mask_i32gather_epi32(vseti(0),mk,idx,sm,sizeof(int));
		sum = _mm512_mask_add_epi32(vseti(0),mk,tem,sum);

		_mm512_mask_i32scatter_epi32(sm, mk, idx, sum, sizeof(int));

		j=_mm512_mask_add_epi32(j,mk,j,vseti(blockDim.x));
		mk=_mm512_mask_cmp_epi32_mask(mk,j,vseti(N),_MM_CMPINT_LT );
		}	
    }

	tmp = _mm512_mask_i32gather_epi32(vseti(0),init_mask,threadIdxx,sm,sizeof(int));
	_mm512_mask_i32scatter_epi32(arr, init_mask, tid, tmp, sizeof(int));

/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void mem_kernel(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old, /*MOD_MAN: Para Begin*/
        int *arr, int N, int *rdom_arr, int *result
 /*MOD_MAN: Para End*/)
{
#include "zpart5.h"
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_mem_kernel *p=(P_PR_mem_kernel *)malloc(num_of_pts_launch*sizeof(P_PR_mem_kernel));//array of object passed to each pthread
    
/*MOD_AUTO*/ //mem_kernel configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_mem_kernel ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    ker_parameters.arr=arr; 
    ker_parameters.N =N;
    ker_parameters.rdom_arr = rdom_arr;
    ker_parameters.result = result;

    int *sm;
    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    if(p[i].warpid==0) {
      sm=(int *)malloc(sizeof(int)*2048);
    }
    p[i].sm=sm;
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for mem_kernel       
    pthread_create(&threads[i], &pt_attr,mem_kernel_KERNEL, (void *)(p+i));        //create with affinity

    #include "zpart9.h"

    /*MOD_MAN*///free shared memory
    for (unsigned i=0; i<num_of_pts_launch; i++) {
        if(p[i].warpid==0){
            //free(p[i].temp_on_cuda); free(p[i].power_on_cuda); free(p[i].temp_t); 
            free(p[i].sm);
        }
    }
    /*MOD_MAN*///Ends

    //Free data
    free(p);free(threads);free(barrier);
}
#endif
