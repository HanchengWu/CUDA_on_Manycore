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
    //int *s_mem;

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

    #include "zpart3.h"    //recover kernel configuration

    /*MOD_MAN*/ //Activate corresponding threadIdx, all vector-lane variant variables
    vint threadIdxx= _mm512_mask_rem_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x)) ;
    vint threadIdxy= _mm512_mask_rem_epi32(vseti(0),init_mask,_mm512_div_epi32(_threadIdx, vseti(blockDim.x)), vseti(blockDim.y)) ;
    vint threadIdxz= _mm512_mask_div_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x*blockDim.y)) ;

    printf("\nI am here!\n");

    for(unsigned _bidx=p->blockgroupid; _bidx<numBlocksPerGrid; _bidx += kp->numActiveBlocks){
        /*MOD_MAN*/ //Activate corresponding blockIdx, all vector-lane invariant variables
        Dim3 blockIdx;
        blockIdx.x = _bidx%gridDim.x; 
        blockIdx.y = (_bidx/gridDim.x)%gridDim.y;
        blockIdx.z = _bidx/(gridDim.x*gridDim.y);
/*MOD_MAN*///KERNEL STARTS//
	vint tem=vseti(0);
        vint tid= _mm512_mask_add_epi32(vseti(0),init_mask,threadIdxx,vseti(blockIdx.x*blockDim.x));
   	_mm512_mask_i32scatter_epi32(result,init_mask,tid,tem,sizeof(int));

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

    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for mem_kernel       
    pthread_create(&threads[i], &pt_attr,mem_kernel_KERNEL, (void *)(p+i));        //create with affinity

    #include "zpart9.h"

    /*MOD_MAN*///free shared memory
    //for (unsigned i=0; i<num_of_pts_launch; i++) {
        //if(p[i].warpid==0){
            //free(p[i].temp_on_cuda); free(p[i].power_on_cuda); free(p[i].temp_t); 
        //}
    //}
    /*MOD_MAN*///Ends

    //Free data
    free(p);free(threads);free(barrier);
}
#endif
