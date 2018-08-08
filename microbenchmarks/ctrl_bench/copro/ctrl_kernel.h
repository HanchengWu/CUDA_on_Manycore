//Replace "ctrl_kernel" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __ctrl_kernel__
#define __ctrl_kernel__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel ctrl_kernel
    //int para;
    int D;
    int *res_arr;
    int costX;

} PR_ctrl_kernel;

//Defines the object passed to each pthread instance for ctrl_kernel
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_ctrl_kernel *kp;//define pointer to PR_ctrl_kernel

} P_PR_ctrl_kernel;

/*MOD_AUTO*/
void *ctrl_kernel_KERNEL(void *arg);

vint cost_func(int costX){
        vint tem = vseti(0);
        vint vec_one = vseti(1);
        for(int i=0;i<costX;++i){
               tem = _mm512_add_epi32(tem,vec_one);
        }
        return tem;
}

#pragma offload_attribute (pop)

void *ctrl_kernel_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_ctrl_kernel *p=(P_PR_ctrl_kernel *)arg;    //restore pthead data pointer
    PR_ctrl_kernel *kp=p->kp;   //restore ctrl_kernel config and para pointer
    
    /*MOD_MAN*///restore kernel ctrl_kernel's parameters
    int D=kp->D;
    int *res_arr=kp->res_arr;
    int costX=kp->costX;
   

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

  vint result=vseti(0);

  for(int i=0; i<D; ++i){ //D’s maximum value is 4
     int tmp=1;
     for (int j=0;j<4-i;++j) tmp=tmp*2;
     int t=16-tmp;

    //if(threadIdxx<16-t){
    vmask mk_if = _mm512_cmp_epi32_mask(threadIdxx, vseti(16-t), _MM_CMPINT_LT);

    result = _mm512_mask_add_epi32(result,mk_if,result, cost_func(costX));

    // }else if (threadIdxx>=16+t){
    vmask mk_else_if = _mm512_mask_cmp_epi32_mask(_mm512_knot(mk_if), threadIdxx, vseti(16+t), _MM_CMPINT_GE );
    //vmask mk_else_if = _mm512_cmp_epi32_mask(threadIdxx, vseti(16+t), _MM_CMPINT_GE );

    result = _mm512_mask_add_epi32(result,mk_else_if,result, cost_func(costX));
    //}

  }
  vint idx=_mm512_add_epi32(threadIdxx,vseti(blockIdx.x*blockDim.x));
  // _mm512_i32scatter_epi32(res_arr,idx,result,sizeof(int));



/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void ctrl_kernel(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old, /*MOD_MAN: Para Begin*/
		int D, int* res_arr, int costX
 /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(usage: ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
		in(D: ONCE) \
		in(res_arr: length(0) REUSE_A) \
		in(costX: ONCE)
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_ctrl_kernel *p=(P_PR_ctrl_kernel *)malloc(num_of_pts_launch*sizeof(P_PR_ctrl_kernel));//array of object passed to each pthread
    
/*MOD_AUTO*/ //ctrl_kernel configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_ctrl_kernel ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    ker_parameters.D = D;
    ker_parameters.res_arr = res_arr;
    ker_parameters.costX= costX; 

    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for ctrl_kernel       
    pthread_create(&threads[i], &pt_attr,ctrl_kernel_KERNEL, (void *)(p+i));        //create with affinity

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

    }//offload ends

}
#endif
