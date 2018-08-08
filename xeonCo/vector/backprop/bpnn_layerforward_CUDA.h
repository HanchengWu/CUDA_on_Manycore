//Replace "bpnn_layerforward_CUDA" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __bpnn_layerforward_CUDA__
#define __bpnn_layerforward_CUDA__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel bpnn_layerforward_CUDA
    //int para;
    float *input_cuda;
    float *output_hidden_cuda;
    float *input_hidden_cuda;
    float *hidden_partial_sum;
    int in; 
    int hid;

} PR_bpnn_layerforward_CUDA;

//Defines the object passed to each pthread instance for bpnn_layerforward_CUDA
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;
    float *input_node;
    float *weight_matrix;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_bpnn_layerforward_CUDA *kp;//define pointer to PR_bpnn_layerforward_CUDA

} P_PR_bpnn_layerforward_CUDA;

/*MOD_AUTO*/
void *bpnn_layerforward_CUDA_KERNEL(void *arg);

__attribute__ (( target(mic) )) int printiv( __m512i v,char *s) {
	int *tem =(int *)&v;
	printf("\n%s:\n",s);
	for (int i=0; i<16;i++,tem++){
    if( *tem != -1 )
				printf("%d, ",*tem);
	}
	printf("\n");
}

__attribute__ (( target(mic) )) int printfv( __m512 v, char *s) {
	float *tem =(float *)&v;
	printf("\n%s:\n",s);
	for (int i=0; i<16;i++,tem++){
				printf("%f, ",*tem);
	}
		printf("\n");
}

#pragma offload_attribute (pop)

void *bpnn_layerforward_CUDA_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_bpnn_layerforward_CUDA *p=(P_PR_bpnn_layerforward_CUDA *)arg;    //restore pthead data pointer
    PR_bpnn_layerforward_CUDA *kp=p->kp;   //restore bpnn_layerforward_CUDA config and para pointer
    float *input_node = p->input_node;
    float *weight_matrix = p->weight_matrix;
    
    /*MOD_MAN*///restore kernel bpnn_layerforward_CUDA's parameters
    //int para=kp->para;
    float *input_cuda = kp->input_cuda;
    float *output_hidden_cuda = kp->output_hidden_cuda;
    float *input_hidden_cuda = kp->input_hidden_cuda;
    float *hidden_partial_sum = kp->hidden_partial_sum;
    int in = kp->in; 
    int hid = kp->hid;


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

//KERNEL STARTS//
         unsigned by = blockIdx.y;
         vint tx = threadIdxx;
         vint ty = threadIdxy;

         //int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
         int index_invariant = (hid+1)*HEIGHT*by + 1 + (hid+1);
         vint index_part2 = _mm512_set1_epi32(hid+1);
         index_part2 = _mm512_mask_mullo_epi32( index_part2, init_mask, index_part2, ty );
         vint index = tx;
         index = _mm512_mask_add_epi32( index, init_mask, index, index_part2 );
         index = _mm512_mask_add_epi32( index, init_mask, index, _mm512_set1_epi32( index_invariant ) );

         //int index_in = HEIGHT * by + ty + 1;
         int index_in_invariant =  HEIGHT * by + 1;
         vint index_in = _mm512_mask_add_epi32( _mm512_set1_epi32(0), init_mask, _mm512_set1_epi32(index_in_invariant), ty );
         vmask M_tx_0 = _mm512_mask_cmp_epi32_mask(init_mask, tx, _mm512_set1_epi32(0), _MM_CMPINT_EQ);
         vfloat input_cuda_index_in = _mm512_mask_i32gather_ps( _mm512_set1_ps(0), M_tx_0, index_in, input_cuda, sizeof(float) );
         _mm512_mask_i32scatter_ps( input_node, M_tx_0, ty, input_cuda_index_in, sizeof(float) );

         pthread_barrier_wait(p->barrier); 
         
         vfloat input_hidden_cuda_index = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), init_mask, index, input_hidden_cuda, sizeof(float) );
         vint ty_tx_index = _mm512_mask_mullo_epi32(_mm512_set1_epi32(-1), init_mask, _mm512_set1_epi32(HEIGHT), ty);
         ty_tx_index = _mm512_mask_add_epi32( ty_tx_index, init_mask, ty_tx_index, tx ); 
         _mm512_mask_i32scatter_ps( weight_matrix, init_mask, ty_tx_index, input_hidden_cuda_index, sizeof(float) );

         pthread_barrier_wait(p->barrier); 

         vfloat w_m_ty_tx1 = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), init_mask, ty_tx_index, weight_matrix, sizeof(float) );
         vfloat i_n_ty = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), init_mask, ty, input_node, sizeof(float) );
         vfloat w_i_mult = _mm512_mask_mul_ps (_mm512_set1_ps(-1), init_mask, w_m_ty_tx1, i_n_ty);
         _mm512_mask_i32scatter_ps( weight_matrix, init_mask, ty_tx_index, w_i_mult, sizeof(float) );

         pthread_barrier_wait(p->barrier); 
            
         for ( int power_two = 2; power_two <= HEIGHT; power_two *= 2 ) {
           vint mod_ty_power_two = _mm512_mask_rem_epi32( _mm512_set1_epi32(-1), init_mask, ty, _mm512_set1_epi32(power_two) );
           vmask M_mod_ty_power_two = _mm512_mask_cmp_epi32_mask( init_mask, mod_ty_power_two, _mm512_set1_epi32(0), _MM_CMPINT_EQ );

           vfloat w_m_ty_tx2 = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), M_mod_ty_power_two, ty_tx_index, weight_matrix, sizeof(float) );

           vint ty_power_two_tx_index = _mm512_mask_add_epi32( _mm512_set1_epi32(-1), M_mod_ty_power_two, ty, _mm512_set1_epi32( power_two/2 ) );
           ty_power_two_tx_index = _mm512_mask_mullo_epi32( _mm512_set1_epi32(-1), M_mod_ty_power_two, ty_power_two_tx_index, _mm512_set1_epi32(HEIGHT) );
           ty_power_two_tx_index = _mm512_mask_add_epi32( _mm512_set1_epi32(-1), M_mod_ty_power_two, ty_power_two_tx_index,  tx );
           vfloat w_m_ty_pow_two = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), M_mod_ty_power_two, ty_power_two_tx_index, weight_matrix, sizeof(float) );

           vfloat weight_added = _mm512_mask_add_ps( _mm512_set1_ps(-1), M_mod_ty_power_two, w_m_ty_tx2, w_m_ty_pow_two );
           _mm512_mask_i32scatter_ps( weight_matrix, M_mod_ty_power_two, ty_tx_index, weight_added, sizeof(float) );

           pthread_barrier_wait(p->barrier); 
         }
            
         vfloat w_m_ty_tx3 = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), init_mask, ty_tx_index, weight_matrix, sizeof(float) );
         _mm512_mask_i32scatter_ps( input_hidden_cuda, init_mask, index, w_m_ty_tx3, sizeof(float) );
         
         pthread_barrier_wait(p->barrier);
         
         vint tx_ty_index = _mm512_mask_mullo_epi32(_mm512_set1_epi32(-1), init_mask, _mm512_set1_epi32(HEIGHT), tx);
         tx_ty_index = _mm512_mask_add_epi32( tx_ty_index, init_mask, tx_ty_index, ty ); 
         vfloat w_m_ty_tx4 = _mm512_mask_i32gather_ps( _mm512_set1_ps(-1), M_tx_0, tx_ty_index, weight_matrix, sizeof(float) );
         
         vint by_hid_ty_index = _mm512_mask_add_epi32( _mm512_set1_epi32(-1), M_tx_0, _mm512_set1_epi32(by*hid), ty );
         _mm512_mask_i32scatter_ps( hidden_partial_sum, M_tx_0, by_hid_ty_index, w_m_ty_tx4, sizeof(float) );
//KERNEL ENDS//
 

/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void bpnn_layerforward_CUDA(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old /*MOD_MAN: Para Begin*/, float *input_cuda, float *output_hidden_cuda, float *input_hidden_cuda, float *hidden_partial_sum, int in, int hid /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(usage:ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
                    in( input_cuda: length(0) REUSE_A ) \
                    in( input_hidden_cuda: length(0) REUSE_A ) \
                    in( output_hidden_cuda: length(0) REUSE_A ) \
                    in( hidden_partial_sum: length(0) REUSE_A ) \
                    in( in: ONCE ) \
                    in( hid: ONCE )
/*MOD_MAN Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN Para*/     //           in( num: ONCE  ) 
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_bpnn_layerforward_CUDA *p=(P_PR_bpnn_layerforward_CUDA *)malloc(num_of_pts_launch*sizeof(P_PR_bpnn_layerforward_CUDA));//array of object passed to each pthread
    
/*MOD_AUTO*/ //bpnn_layerforward_CUDA configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_bpnn_layerforward_CUDA ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    //ker_parameters.arr=arr; 
    ker_parameters.input_cuda = input_cuda;
    ker_parameters.output_hidden_cuda = output_hidden_cuda;
    ker_parameters.input_hidden_cuda = input_hidden_cuda;
    ker_parameters.hidden_partial_sum = hidden_partial_sum;
    ker_parameters.in = in; 
    ker_parameters.hid = hid;

    float *input_node;
    float *weight_matrix;


    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED

    if(p[i].warpid==0) {
      //p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
      input_node = (float *)malloc(sizeof(float)*HEIGHT);
      weight_matrix = (float *)malloc(HEIGHT*WIDTH*sizeof(float));
    }

    p[i].input_node = input_node;
    p[i].weight_matrix = weight_matrix;

   //SHARED MEM ENDS
    
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for bpnn_layerforward_CUDA       
    pthread_create(&threads[i], &pt_attr,bpnn_layerforward_CUDA_KERNEL, (void *)(p+i));        //create with affinity

    #include "zpart9.h"

    for ( int i = 0; i < num_of_pts_launch; i++ ) {
      if (p[i].warpid == 0) {
        free(p[i].input_node);
        free(p[i].weight_matrix);
      }
    }
    
    free(p); free(threads); free(barrier);
    }//offload ends

}
#endif
