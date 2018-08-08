//Replace "Kernel2" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __Kernel2__
#define __Kernel2__

//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel Kernel2
    //int para;
    int* g_graph_mask;
    int* g_updating_graph_mask;
    int* g_graph_visited;
    int* g_over;
    int no_of_nodes;
} PR_Kernel2;

//Defines the object passed to each pthread instance for Kernel2
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_Kernel2 *kp;//define pointer to PR_Kernel2

} P_PR_Kernel2;

/*MOD_AUTO*/
void *Kernel2_KERNEL(void *arg);


void *Kernel2_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_Kernel2 *p=(P_PR_Kernel2 *)arg;    //restore pthead data pointer
    PR_Kernel2 *kp=p->kp;   //restore Kernel2 config and para pointer
    
    /*MOD_MAN*///restore kernel Kernel2's parameters
    //int para=kp->para;
    int* g_graph_mask = kp->g_graph_mask;
    int* g_updating_graph_mask = kp->g_updating_graph_mask;
    int* g_graph_visited = kp->g_graph_visited;
    int* g_over = kp->g_over;
    int no_of_nodes = kp->no_of_nodes;

    #include "zpart3.h"    //recover kernel configuration

    /*MOD_MAN*/ //Activate corresponding threadIdx, all vector-lane variant variables
    vint threadIdxx= _mm512_mask_rem_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x)) ;
    vint threadIdxy= _mm512_mask_rem_epi32(vseti(0),init_mask,_mm512_div_epi32(_threadIdx, vseti(blockDim.x)), vseti(blockDim.y)) ;
    vint threadIdxz= _mm512_mask_div_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x*blockDim.y)) ;

    for(unsigned _bidx=p->blockgroupid; _bidx<numBlocksPerGrid; _bidx += kp->numActiveBlocks){
        /*MOD_MAN*/ //Activate corresponding blockIdx, all vector-lane invariant variables
        Dim3 blockIdx;
        blockIdx.x = _bidx%gridDim.x; 
        //blockIdx.y = (_bidx/gridDim.x)%gridDim.y;
        //blockIdx.z = _bidx/(gridDim.x*gridDim.y);
/*MOD_MAN*///KERNEL STARTS//

    //int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
    vint tid = _mm512_mask_add_epi32( _mm512_set1_epi32(0), init_mask, _mm512_set1_epi32(blockIdx.x*16), threadIdxx ); 

    // if( tid<no_of_nodes ) 
    vmask M_if_relevant = _mm512_cmp_epi32_mask( tid, _mm512_set1_epi32(no_of_nodes), _MM_CMPINT_LT );

    // if( tid<no_of_nodes && g_updating_graph_mask[tid]) 
    vint g_updating_graph_mask_temp = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(0), M_if_relevant, tid, g_updating_graph_mask, sizeof(int) );
    vmask M_if_updating_graph_mask = _mm512_mask_cmp_epi32_mask( M_if_relevant, g_updating_graph_mask_temp, _mm512_set1_epi32(1), _MM_CMPINT_EQ );

    // g_graph_mask[tid] = true;
    _mm512_mask_i32scatter_epi32( g_graph_mask, M_if_updating_graph_mask, tid, _mm512_set1_epi32(1), sizeof(int) );

    // g_graph_visited[tid]=true;
    _mm512_mask_i32scatter_epi32( g_graph_visited, M_if_updating_graph_mask, tid, _mm512_set1_epi32(1), sizeof(int) );

    // TODO: which is fater?
    if ( _mm512_mask2int( M_if_updating_graph_mask ) != 0 ) {
      *g_over = 1;
    }
    //g_over = _mm512_mask_add_epi32( _mm512_set1_epi32(0), M_if_updating_graph_mask, _mm512_set1_epi32(1), _mm512_set1_epi32(0) );

    // g_updating_graph_mask[tid]=false;
    _mm512_mask_i32scatter_epi32( g_updating_graph_mask, M_if_updating_graph_mask, tid, _mm512_set1_epi32(0), sizeof(int) );





/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void Kernel2(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old /*MOD_MAN: Para Begin*/ , int *d_graph_mask, int *d_updating_graph_mask, int *d_graph_visited, int *d_over, int no_of_nodes  /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_Kernel2 *p=(P_PR_Kernel2 *)malloc(num_of_pts_launch*sizeof(P_PR_Kernel2));//array of object passed to each pthread
    
/*MOD_AUTO*/ //Kernel2 configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_Kernel2 ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    //ker_parameters.arr=arr; 
    ker_parameters.g_graph_mask = d_graph_mask;
    ker_parameters.g_updating_graph_mask = d_updating_graph_mask;
    ker_parameters.g_graph_visited = d_graph_visited;
    ker_parameters.g_over = d_over;
    ker_parameters.no_of_nodes = no_of_nodes;
    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for Kernel2       
    pthread_create(&threads[i], &pt_attr,Kernel2_KERNEL, (void *)(p+i));        //create with affinity

    #include "zpart9.h"

    free(p); free(threads); free(barrier);

}
#endif
