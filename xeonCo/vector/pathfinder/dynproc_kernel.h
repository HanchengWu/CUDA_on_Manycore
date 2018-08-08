//Replace "dynproc_kernel" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __dynproc_kernel__
#define __dynproc_kernel__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
#define BLOCK_SIZE 256
#define STR_SIZE 256
#define HALO 1 // halo width along one direction when advancing to the next iteration


typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel dynproc_kernel
    //int para;
    int iteration;
    int *gpuWall;
    int *gpuSrc;
    int *gpuResults;
    int cols;
    int rows;
    int startStep;
    int border;

} PR_dynproc_kernel;

//Defines the object passed to each pthread instance for dynproc_kernel 
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;
    int *s_prev;
    int *s_result;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_dynproc_kernel *kp;//define pointer to PR_dynproc_kernel

} P_PR_dynproc_kernel;

/*MOD_AUTO*/
void *dynproc_kernel_KERNEL(void *arg);

__attribute__ (( target(mic) )) int printiv( __m512i v,char *s) {
  int *tem =(int *)&v;
  printf("\n%s:\n",s);
  for (int i=0; i<16;i++,tem++){
    if( *tem != -1 )
      printf("%d, ",*tem);
  }
  printf("\n");
}

#pragma offload_attribute (pop)

void *dynproc_kernel_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_dynproc_kernel *p=(P_PR_dynproc_kernel *)arg;    //restore pthead data pointer
    PR_dynproc_kernel *kp=p->kp;   //restore dynproc_kernel config and para pointer
    
    /*MOD_MAN*///restore kernel dynproc_kernel's parameters
    //int para=kp->para;
    int iteration = kp->iteration;
    int *gpuWall = kp->gpuWall;
    int *gpuSrc = kp->gpuSrc;
    int *gpuResults = kp->gpuResults;
    int cols = kp->cols;
    int rows = kp->rows;
    int startStep = kp->startStep;
    int border = kp->border;

    int *prev = p->s_prev;
    int *result = p->s_result;

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

        vint tx = threadIdxx;

        int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        int blkX = small_block_cols*blockIdx.x-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        vint xidx = _mm512_mask_add_epi32 (_mm512_set1_epi32(0), init_mask, _mm512_set1_epi32(blkX), tx);

        vint W = _mm512_mask_add_epi32 (_mm512_set1_epi32(0), init_mask, tx, _mm512_set1_epi32(-1));

        vint E = _mm512_mask_add_epi32 (_mm512_set1_epi32(0), init_mask, tx, _mm512_set1_epi32(1));

        vmask M_w_less = _mm512_mask_cmp_epi32_mask (init_mask, W, _mm512_set1_epi32(validXmin), _MM_CMPINT_LT);
        W = _mm512_mask_add_epi32 (W, M_w_less, _mm512_set1_epi32(validXmin), _mm512_set1_epi32(0));

        vmask M_e_less = _mm512_mask_cmp_epi32_mask (init_mask, E, _mm512_set1_epi32(validXmax), _MM_CMPINT_GT);
        E = _mm512_mask_add_epi32 (E, M_e_less, _mm512_set1_epi32(validXmax), _mm512_set1_epi32(0));

        vmask M_isValid = _mm512_mask_cmp_epi32_mask(init_mask, tx, _mm512_set1_epi32(validXmin), _MM_CMPINT_GE);
        M_isValid = _mm512_mask_cmp_epi32_mask( M_isValid, tx, _mm512_set1_epi32(validXmax), _MM_CMPINT_LE);

        vmask M_x_inRange = _mm512_mask_cmp_epi32_mask( init_mask, xidx, _mm512_set1_epi32(0), _MM_CMPINT_GE);
        M_x_inRange = _mm512_mask_cmp_epi32_mask( M_x_inRange, xidx, _mm512_set1_epi32(cols-1), _MM_CMPINT_LE);

        vint gpuSrc_xidx = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(0), M_x_inRange, xidx, gpuSrc, sizeof(int) ); 
        //printiv(tx, "tx");
        //fflush(0);
        _mm512_mask_i32scatter_epi32( prev, M_x_inRange, tx, gpuSrc_xidx, sizeof(int) );

         pthread_barrier_wait(p->barrier); 

        vmask M_computed;

        for (int ii=0; ii<iteration ; ii++){
              M_computed = _mm512_int2mask(0);

              vmask M_tx_valid = _mm512_cmp_epi32_mask( tx, _mm512_set1_epi32(ii+1), _MM_CMPINT_GE );
              M_tx_valid = _mm512_mask_cmp_epi32_mask( M_tx_valid, tx, _mm512_set1_epi32( BLOCK_SIZE-ii-2 ), _MM_CMPINT_LE );
              M_tx_valid = _mm512_kand(M_tx_valid, M_isValid);

              M_computed = M_tx_valid;

              vint left = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(-1), M_tx_valid, W, prev, sizeof(int) ); 

              vint up = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(-1), M_tx_valid, tx, prev, sizeof(int) ); 

              vint right = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(-1), M_tx_valid, E, prev, sizeof(int) ); 

              vmask M_shortest_l_u = _mm512_mask_cmp_epi32_mask( M_tx_valid, left, up, _MM_CMPINT_LE );
              vint shortest = _mm512_mask_add_epi32( up, M_shortest_l_u, left, _mm512_set1_epi32(0) );

              vmask M_shortest_s_r = _mm512_mask_cmp_epi32_mask( M_tx_valid, shortest, right, _MM_CMPINT_LE );
              shortest = _mm512_mask_add_epi32( right, M_shortest_s_r, shortest, _mm512_set1_epi32(0) );

              vint index = _mm512_mask_add_epi32 ( _mm512_set1_epi32(0), M_tx_valid, _mm512_set1_epi32(cols*(startStep+ii)), xidx);

              vint shortest_gpuWall = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(0), M_tx_valid, index, gpuWall, sizeof(int) );
              shortest_gpuWall = _mm512_mask_add_epi32( _mm512_set1_epi32(0), M_tx_valid, shortest_gpuWall, shortest );
              _mm512_mask_i32scatter_epi32( result, M_tx_valid, tx, shortest_gpuWall, sizeof(int) );

              pthread_barrier_wait(p->barrier); 

              if(ii==iteration-1)
                break;

              vint result_tx = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(0), M_computed, tx, result, sizeof(int) );
              _mm512_mask_i32scatter_epi32( prev, M_computed, tx, result_tx, sizeof(int) );
 
              pthread_barrier_wait(p->barrier); 
          }

            vint result_tx = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(0), M_computed, tx, result, sizeof(int) );
            _mm512_mask_i32scatter_epi32( gpuResults, M_computed, xidx, result_tx, sizeof(int) );
 

/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void dynproc_kernel(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old /*MOD_MAN: Para Begin*/, int iteration, int *gpuWall, int *gpuSrc, int *gpuResults, int cols, int rows, int startStep, int border /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(usage:ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
                                     in( iteration: ONCE ) \
                                     in( gpuWall: length(0) REUSE_A ) \
                                     in( gpuSrc: length(0) REUSE_A ) \
                                     in( gpuResults: length(0) REUSE_A ) \
                                     in( cols: ONCE ) \
                                     in( rows: ONCE ) \
                                     in( startStep: ONCE ) \
                                     in( border: ONCE ) 
/*MOD_MAN Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN Para*/     //           in( num: ONCE  ) 
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_dynproc_kernel *p=(P_PR_dynproc_kernel *)malloc(num_of_pts_launch*sizeof(P_PR_dynproc_kernel));//array of object passed to each pthread
    
/*MOD_AUTO*/ //dynproc_kernel configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_dynproc_kernel ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    //ker_parameters.arr=arr; 
    ker_parameters.iteration = iteration;
    ker_parameters.gpuWall = gpuWall;
    ker_parameters.gpuSrc = gpuSrc;
    ker_parameters.gpuResults = gpuResults;
    ker_parameters.cols = cols;
    ker_parameters.rows = rows;
    ker_parameters.startStep = startStep;
    ker_parameters.border = border;


    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    int *s_prev;
    int *s_result;

    if(p[i].warpid==0) {
      s_prev=(int *)malloc(sizeof(int)*BLOCK_SIZE);
      s_result=(int *)malloc(sizeof(int)*BLOCK_SIZE);
    }

    p[i].s_prev = s_prev;
    p[i].s_result = s_result;

    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for dynproc_kernel       
    pthread_create(&threads[i], &pt_attr,dynproc_kernel_KERNEL, (void *)(p+i));        //create with affinity
    
    #include "zpart9.h"

    for ( int i = 0; i < num_of_pts_launch; i++ ) {
      if (p[i].warpid == 0) {
        free(p[i].s_prev);
        free(p[i].s_result);
      }
    }
    
    free(p); free(threads); free(barrier);
    }//offload ends

}
#endif
