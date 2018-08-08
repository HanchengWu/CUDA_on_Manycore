//Replace "bpnn_adjust_weights_cuda" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __bpnn_adjust_weights_cuda__
#define __bpnn_adjust_weights_cuda__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel bpnn_adjust_weights_cuda 
    //int para;
    float *delta; 
    int hid; 
    float *ly;
    int in; 
    float *w; 
    float *oldw;

} PR_bpnn_adjust_weights_cuda;

//Defines the object passed to each pthread instance for bpnn_adjust_weights_cuda
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_bpnn_adjust_weights_cuda *kp;//define pointer to PR_bpnn_adjust_weights_cuda

} P_PR_bpnn_adjust_weights_cuda;

/*MOD_AUTO*/
void *bpnn_adjust_weights_cuda_KERNEL(void *arg);

#pragma offload_attribute (pop)

void *bpnn_adjust_weights_cuda_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_bpnn_adjust_weights_cuda *p=(P_PR_bpnn_adjust_weights_cuda *)arg;    //restore pthead data pointer
    PR_bpnn_adjust_weights_cuda *kp=p->kp;   //restore bpnn_adjust_weights_cuda config and para pointer
    
    /*MOD_MAN*///restore kernel bpnn_adjust_weights_cuda's parameters
    //int para=kp->para;
    float *delta = kp->delta; 
    int hid = kp->hid; 
    float *ly = kp->ly;
    int in = kp->in; 
    float *w = kp->w; 
    float *oldw = kp->oldw;

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

      unsigned by = blockIdx.y;
      vint tx = threadIdxx;
      vint ty = threadIdxy;

//    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
      int index_invariant = (hid+1)*HEIGHT*by + 1 + (hid+1);
      vint index_part2 = _mm512_set1_epi32(hid+1);
      index_part2 = _mm512_mask_mullo_epi32( index_part2, init_mask, index_part2, ty );
      vint index = tx;
      index = _mm512_mask_add_epi32( index, init_mask, index, index_part2 );
      index = _mm512_mask_add_epi32( index, init_mask, index, _mm512_set1_epi32( index_invariant ) );

//    int index_y = HEIGHT * by + ty + 1;
      int index_y_invariant =  HEIGHT * by + 1;
      vint index_y = _mm512_mask_add_epi32( _mm512_set1_epi32(-1), init_mask, _mm512_set1_epi32(index_y_invariant), ty );

//    int index_x = tx + 1;
      vint index_x = _mm512_mask_add_epi32( _mm512_set1_epi32(-1), init_mask, tx, _mm512_set1_epi32(1) );

//    w[index] +=
//        ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      vfloat w_index = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), init_mask, index, w, sizeof(float) );
      vfloat delta_index_x = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), init_mask, index_x, delta, sizeof(float) );
      vfloat ly_index_y = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), init_mask, index_y, ly, sizeof(float) );
      vfloat oldw_index = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), init_mask, index, oldw, sizeof(float) );
      vfloat delta_ETA = _mm512_mask_mul_ps( _mm512_set1_ps(-1), init_mask, delta_index_x, _mm512_set1_ps(ETA) );
      vfloat w_oldw_MOM = _mm512_mask_mul_ps( _mm512_set1_ps(-1), init_mask, oldw_index, _mm512_set1_ps(MOMENTUM) );
      vfloat oldw_index_final = _mm512_mask_mul_ps( _mm512_set1_ps(-1), init_mask, delta_ETA, ly_index_y );
      oldw_index_final = _mm512_mask_add_ps( oldw_index_final, init_mask, oldw_index_final, w_oldw_MOM );
      vfloat w_index_final = _mm512_mask_add_ps( _mm512_set1_ps(-1), init_mask, oldw_index_final, w_index );
      _mm512_mask_i32scatter_ps( w, init_mask, index, w_index_final, sizeof(float) );

//    oldw[index] =
//        ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      _mm512_mask_i32scatter_ps( oldw, init_mask, index, oldw_index_final, sizeof(float) );

//    __syncthreads();
      pthread_barrier_wait(p->barrier); 

//    if (ty == 0 && by == 0) {
      vmask M_ty_0 = _mm512_mask_cmp_epi32_mask(init_mask, ty, _mm512_set1_epi32(0), _MM_CMPINT_EQ);
      vmask M_ty_by_0 = _mm512_mask_cmp_epi32_mask(M_ty_0, _mm512_set1_epi32(by), _mm512_set1_epi32(0), _MM_CMPINT_EQ);

//        w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
//        oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));

      vfloat w_index_x = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), M_ty_by_0, index_x, w, sizeof(float) );
      vfloat delta_index_x2 = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), M_ty_by_0, index_x, delta, sizeof(float) );
      vfloat oldw_index_x = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), M_ty_by_0, index_x, oldw, sizeof(float) );
      vfloat delta_ETA2 = _mm512_mask_mul_ps( _mm512_set1_ps(-1), M_ty_by_0, delta_index_x2, _mm512_set1_ps(ETA) );
      vfloat w_oldw_x_MOM = _mm512_mask_mul_ps( _mm512_set1_ps(-1), M_ty_by_0, oldw_index_x, _mm512_set1_ps(MOMENTUM) );
      vfloat oldw_index_x_final = _mm512_mask_add_ps( _mm512_set1_ps(-1), M_ty_by_0, delta_ETA2, w_oldw_x_MOM );
      vfloat w_index_x_final = _mm512_mask_add_ps( _mm512_set1_ps(-1), M_ty_by_0, oldw_index_x_final, w_index_x );
      _mm512_mask_i32scatter_ps( w, M_ty_by_0, index_x, w_index_x_final, sizeof(float) );
      _mm512_mask_i32scatter_ps( oldw, M_ty_by_0, index_x, oldw_index_x_final, sizeof(float) );

//    }

/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void bpnn_adjust_weights_cuda(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old /*MOD_MAN: Para Begin*/, float *delta, int hid, float *ly, int in, float *w, float *oldw /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(usage:ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
                            in( delta: length(0) REUSE_A ) \
                            in( hid: ONCE ) \
                            in( ly: length(0) REUSE_A ) \
                            in( in: ONCE ) \
                            in( w: length(0) REUSE_A ) \
                            in( oldw: length(0) REUSE_A )
/*MOD_MAN Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN Para*/     //           in( num: ONCE  ) 
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_bpnn_adjust_weights_cuda *p=(P_PR_bpnn_adjust_weights_cuda *)malloc(num_of_pts_launch*sizeof(P_PR_bpnn_adjust_weights_cuda));//array of object passed to each pthread
    
/*MOD_AUTO*/ //bpnn_adjust_weights_cuda configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_bpnn_adjust_weights_cuda ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    //ker_parameters.arr=arr; 
    ker_parameters.delta = delta;
    ker_parameters.hid = hid;
    ker_parameters.ly = ly;
    ker_parameters.in = in;
    ker_parameters.w = w;
    ker_parameters.oldw = oldw;
 
    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for bpnn_adjust_weights_cuda       
    pthread_create(&threads[i], &pt_attr,bpnn_adjust_weights_cuda_KERNEL, (void *)(p+i));        //create with affinity

    #include "zpart9.h"

    free(p); free(threads); free(barrier);

    }//offload ends

}
#endif
