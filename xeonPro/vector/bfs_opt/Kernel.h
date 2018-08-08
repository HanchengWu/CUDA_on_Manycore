//Replace "Kernel" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __Kernel__
#define __Kernel__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel Kernel
    //int para;

    Node* g_graph_nodes;
    int* g_graph_edges;
    int* g_graph_mask;
    int* g_updating_graph_mask;
    int* g_graph_visited;
    int* g_cost;
    int no_of_nodes;

} PR_Kernel;

//Defines the object passed to each pthread instance for Kernel
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_Kernel *kp;//define pointer to PR_Kernel

} P_PR_Kernel;

/*MOD_AUTO*/
void *Kernel_KERNEL(void *arg);

#pragma offload_attribute (pop)

void *Kernel_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_Kernel *p=(P_PR_Kernel *)arg;    //restore pthead data pointer
    PR_Kernel *kp=p->kp;   //restore Kernel config and para pointer
    
    /*MOD_MAN*///restore kernel Kernel's parameters
    //int para=kp->para;
    Node* g_graph_nodes = kp->g_graph_nodes;
    int* g_graph_edges = kp->g_graph_edges;
    int* g_graph_mask = kp->g_graph_mask;
    int* g_updating_graph_mask = kp->g_updating_graph_mask;
    int* g_graph_visited = kp->g_graph_visited;
    int* g_cost = kp->g_cost;
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
/*
        vint tid = _mm512_mask_add_epi32( _mm512_set1_epi32(0), init_mask, _mm512_set1_epi32(blockIdx.x*16), threadIdxx ); 

        vmask M_if_relevant = _mm512_mask_cmp_epi32_mask( init_mask, tid, _mm512_set1_epi32(no_of_nodes), _MM_CMPINT_LT );

        vint g_graph_mask_temp = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(0), M_if_relevant, tid, g_graph_mask, sizeof(int) );
        vmask M_if_graph_mask = _mm512_mask_cmp_epi32_mask( M_if_relevant, g_graph_mask_temp, _mm512_set1_epi32(1), _MM_CMPINT_EQ );

        _mm512_mask_i32scatter_epi32( g_graph_mask, M_if_relevant, tid, _mm512_set1_epi32(0), sizeof(int) );

        vint g_graph_nodes_starting_temp = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(0), M_if_graph_mask, tid, &g_graph_nodes[0].starting, sizeof(Node) );

        vint i_frontier = _mm512_mask_add_epi32( _mm512_set1_epi32(-1), M_if_graph_mask, _mm512_set1_epi32(0), g_graph_nodes_starting_temp ); 
        vint g_graph_nodes_no_of_edges_temp = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(0), M_if_graph_mask, tid, &g_graph_nodes[0].no_of_edges, sizeof(Node) );
        vmask M_i_frontier = _mm512_mask_cmp_epi32_mask( M_if_graph_mask, i_frontier, _mm512_add_epi32(g_graph_nodes_no_of_edges_temp, g_graph_nodes_starting_temp), _MM_CMPINT_LT ); // cond

        while ( _mm512_mask2int(M_i_frontier) != 0 ) {
          vint id = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(-1), M_i_frontier, i_frontier, g_graph_edges, sizeof(int) ); 
          
          vint id_graph_visited = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(-1), M_i_frontier, id, g_graph_visited, sizeof(int) );
          vmask M_graph_visited = _mm512_mask_cmp_epi32_mask( M_i_frontier, id_graph_visited, _mm512_set1_epi32(0), _MM_CMPINT_EQ );

        vint g_cost_temp = _mm512_mask_i32gather_epi32( _mm512_set1_epi32(0), init_mask, tid, g_cost, sizeof(int) ); 
          vint g_costs = _mm512_mask_add_epi32( _mm512_set1_epi32(-1), M_graph_visited, g_cost_temp, _mm512_set1_epi32(1) );
          _mm512_mask_i32scatter_epi32( g_cost, M_graph_visited, id, g_costs, sizeof(int) );

          _mm512_mask_i32scatter_epi32( g_updating_graph_mask, M_graph_visited, id, _mm512_set1_epi32(1), sizeof(int) );

          i_frontier = _mm512_mask_add_epi32(i_frontier, M_i_frontier, i_frontier, _mm512_set1_epi32(1)); // increment
          M_i_frontier = _mm512_mask_cmp_epi32_mask( M_i_frontier, i_frontier, _mm512_add_epi32(g_graph_nodes_no_of_edges_temp, g_graph_nodes_starting_temp), _MM_CMPINT_LT ); // cond
        }

*/
/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void Kernel(unsigned numPtsPerCore, const dim3 &dimGrid_old, const dim3 &dimBlock_old /*MOD_MAN: Para Begin*/ , Node *d_graph_nodes, int *d_graph_edges, int *d_graph_mask, int *d_updating_graph_mask, int *d_graph_visited, int *d_cost, int no_of_nodes /*MOD_MAN: Para End*/)
{

//PART5 starts
Dim3 dimGrid,dimBlock;
dimGrid.x=dimGrid_old.x; dimGrid.y=dimGrid_old.y; dimGrid.z=dimGrid_old.z;
dimBlock.x=dimBlock_old.x; dimBlock.y=dimBlock_old.y; dimBlock.z=dimBlock_old.z;
//PART5 ends

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
            in( d_graph_nodes: length(0) REUSE_A ) \
            in( d_graph_edges: length(0) REUSE_A ) \
            in( d_graph_mask: length(0) REUSE_A ) \
            in( d_updating_graph_mask: length(0) REUSE_A ) \
            in( d_graph_visited: length(0) REUSE_A ) \
            in( d_cost: length(0) REUSE_A ) \
            in( no_of_nodes: ONCE )
/*MOD_MAN Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN Para*/     //           in( num: ONCE  ) 
    {//offload begins

    //Below calculates pthreads configuration
//PART6 starts
    //Below calculates pthreads configuration
    unsigned AvaiLogicCore=sysconf(_SC_NPROCESSORS_ONLN)-4;//get # of availabe logical cores, avoid the last 4(spare the last phisical core for OS&I/O)
    set_self_affixed_to_spare_core(sysconf(_SC_NPROCESSORS_ONLN)-1);

    if(DEBUG) {
        printf("\nNum of Available Logical Processors on Phi: %d. \n",AvaiLogicCore);
        printf("\nXeon Phi Master Thread. Runing on Logical Core: %d.\n",sched_getcpu()); fflush(0);
    }
    //BELOW PART CALCULATES DEVICE UTILIZATION AND PTHREAD CONFIGURATION
    unsigned AvaiNumPts=AvaiLogicCore/4*numPtsPerCore;//get # of availabe pthreads
    unsigned numTdsPerBlock=dimBlock.x*dimBlock.y*dimBlock.z;
    unsigned PtsPerBlock=numTdsPerBlock/16; if(0!=numTdsPerBlock%16) PtsPerBlock++;//# of pthreads assigned to a block based on cuThreads
    //unsigned maxConcBlocks=AvaiNumPts/PtsPerBlock; //Max num of concurrently running cuBlocks 
    unsigned maxConcBlocks=AvaiNumPts/PtsPerBlock*0.5; //Max num of concurrently running cuBlocks 
    if (maxConcBlocks==0) maxConcBlocks=1;

    unsigned num_of_pts_launch,numActiveBlocks;
    unsigned numBlksPerGrid = dimGrid.x*dimGrid.y*dimGrid.z;
 
    if(maxConcBlocks>=numBlksPerGrid){ //launch smaller between maxConcBlocks and numCuBlocks
        num_of_pts_launch=numBlksPerGrid*PtsPerBlock;
        numActiveBlocks=numBlksPerGrid;
    }else{
        //This indicates pthreads grouped in blocks have to iteration to finish numCuBlocks
        num_of_pts_launch=maxConcBlocks*PtsPerBlock;
        numActiveBlocks=maxConcBlocks;
    }
    if(0==numActiveBlocks) { printf("\nBlock is configured too large to be put on Xeon phi or Congiured as 0 block.\n"); fflush(0);}
    //DONE

    //Create data hole pthreads ids, barriers and attibute
    pthread_t *threads=(pthread_t *)malloc(num_of_pts_launch*sizeof(pthread_t));//pthread id array
    pthread_attr_t pt_attr; pthread_attr_init(&pt_attr);//init pthread attribute
    pthread_barrier_t *barrier=(pthread_barrier_t *)malloc(numActiveBlocks*sizeof(pthread_barrier_t));//pthreads grouped into the same block will share a barrier
//PART6 ends

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_Kernel *p=(P_PR_Kernel *)malloc(num_of_pts_launch*sizeof(P_PR_Kernel));//array of object passed to each pthread
    
/*MOD_AUTO*/ //Kernel configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_Kernel ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    //ker_parameters.arr=arr; 
    ker_parameters.g_graph_nodes = d_graph_nodes;
    ker_parameters.g_graph_edges = d_graph_edges;
    ker_parameters.g_graph_mask = d_graph_mask;
    ker_parameters.g_updating_graph_mask = d_updating_graph_mask;
    ker_parameters.g_graph_visited = d_graph_visited;
    ker_parameters.g_cost = d_cost;
    ker_parameters.no_of_nodes = no_of_nodes;
    
//PART7 starts

//init each barrier to sync PtsPerBlock pthreads, PtsPerBlock is the num of pthreads that make up a Cuda Block
    for(unsigned i=0;i<numActiveBlocks;++i){  //So, each barrier acts as a __synchthreads()
        pthread_barrier_init(barrier+i,NULL,PtsPerBlock);
    }

    cpu_set_t phi_set;//phi_set used to set pthread affinity
    unsigned j=0; unsigned core=0; unsigned gap=4-numPtsPerCore;

    for (unsigned i=0; i<num_of_pts_launch; i++) {
        //Init parameters for pthread/cudaBlock
        p[i].warpid=i%PtsPerBlock; //set pthread's warpid,recall that each warp here has 16 lanes
//PART7 ends

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for Kernel       
    pthread_create(&threads[i], &pt_attr,Kernel_KERNEL, (void *)(p+i));        //create with affinity

    #include "zpart9.h"

    free(p); free(threads); free(barrier);
    }//offload ends

}
#endif
