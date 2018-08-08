//Replace "KNAME" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __KNAME__
#define __KNAME__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel KNAME
    //int para;

} PR_KNAME;

//Defines the object passed to each pthread instance for KNAME
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_KNAME *kp;//define pointer to PR_KNAME

} P_PR_KNAME;

/*MOD_AUTO*/
void *KNAME_KERNEL(void *arg);

#pragma offload_attribute (pop)

void *KNAME_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_KNAME *p=(P_PR_KNAME *)arg;    //restore pthead data pointer
    PR_KNAME *kp=p->kp;   //restore KNAME config and para pointer
    
    /*MOD_MAN*///restore kernel KNAME's parameters
    //int para=kp->para;
   

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





/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void KNAME(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old /*MOD_MAN: Para Begin*/ /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(usage: ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
/*MOD_MAN Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN Para*/     //           in( num: ONCE  ) 
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_KNAME *p=(P_PR_KNAME *)malloc(num_of_pts_launch*sizeof(P_PR_KNAME));//array of object passed to each pthread
    
/*MOD_AUTO*/ //KNAME configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_KNAME ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    //ker_parameters.arr=arr; 

    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for KNAME       
    pthread_create(&threads[i], &pt_attr,KNAME_KERNEL, (void *)(p+i));        //create with affinity

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
