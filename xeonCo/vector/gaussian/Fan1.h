//Replace "Fan1" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __Fan1__
#define __Fan1__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel Fan1
    float *m_cuda;
    float *a_cuda;
    int Size;
    int t;
    //int para;

} PR_Fan1;

//Defines the object passed to each pthread instance for Fan1
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_Fan1 *kp;//define pointer to PR_Fan1

} P_PR_Fan1;

/*MOD_AUTO*/
void *Fan1_KERNEL(void *arg);

#pragma offload_attribute (pop)

void *Fan1_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_Fan1 *p=(P_PR_Fan1 *)arg;    //restore pthead data pointer
    PR_Fan1 *kp=p->kp;   //restore Fan1 config and para pointer
    
    /*MOD_MAN*///restore kernel Fan1's parameters
    float *m_cuda=kp->m_cuda;
    float *a_cuda=kp->a_cuda;
    int Size=kp->Size;
    int t=kp->t;
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

    //if(threadIdx.x + blockIdx.x * blockDim.x < Size-1-t) {
        vint gidx= _mm512_mask_add_epi32(vseti(0),init_mask,vseti(blockDim.x*blockIdx.x),threadIdxx);

        vmask mif_t=_mm512_mask_cmp_epi32_mask(init_mask, gidx, vseti(Size-1-t), _MM_CMPINT_LT);
        //*(m_cuda+Size*(gidx+t+1)+t) = *(a_cuda+Size*(gidx+t+1)+t) / *(a_cuda+Size*t+t);
        //m_cuda[Size*(gidx+t+1)+t] = a_cuda[Size*(gidx+t+1)+t] / a_cuda[Size*t+t] ;
        //=>
        //vfloat gdixt = gidx+t+1;
        //m_cuda[Size*gdixt+t] = a_cuda[Size*gdixt+t] / a_cuda[Size*t+t] ;
            vint gidxt =_mm512_mask_add_epi32(vseti(0),mif_t,gidx,vseti(t+1));
            //calculate right value
            vint rpidx = _mm512_mask_mullo_epi32(vseti(0), mif_t, vseti(Size),gidxt);
            rpidx = _mm512_mask_add_epi32(vseti(0),mif_t,rpidx,vseti(t));
            vfloat vrval=_mm512_mask_i32gather_ps(vrval,mif_t,rpidx,a_cuda,sizeof(float));
            vrval = _mm512_mask_div_ps(vrval,mif_t,vrval, vsetf(a_cuda[Size*t+t]));
            //commit to left
            _mm512_mask_i32scatter_ps(m_cuda, mif_t, rpidx, vrval, sizeof(float));
    //}

/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void Fan1(unsigned numPtsPerCore, float usage,const dim3 &dimGrid_old, const dim3 &dimBlock_old, 
                        /*MOD_MAN: Para Begin*/ 
                        float *m_cuda, float *a_cuda, int Size, int t
                        /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(usage: ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
                                    in(m_cuda: length(0) REUSE_A) \
                                    in(a_cuda: length(0) REUSE_A) \
                                    in(Size: ONCE) \
                                    in(t: ONCE)                                    
/*MOD_MAN Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN Para*/     //           in( num: ONCE  ) 
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_Fan1 *p=(P_PR_Fan1 *)malloc(num_of_pts_launch*sizeof(P_PR_Fan1));//array of object passed to each pthread
    
/*MOD_AUTO*/ //Fan1 configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_Fan1 ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    ker_parameters.m_cuda = m_cuda;
    ker_parameters.a_cuda = a_cuda;
    ker_parameters.Size = Size;
    ker_parameters.t = t;
    //ker_parameters.arr=arr; 

    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for Fan1       
    pthread_create(&threads[i], &pt_attr,Fan1_KERNEL, (void *)(p+i));        //create with affinity

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
