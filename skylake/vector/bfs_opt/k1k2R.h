//Replace "k1k2R" with your "k1k2R NAME", then modify places marked with "MOD_MAN"
#ifndef __k1k2R__
#define __k1k2R__

//Struct holds kernel config and info used for synchronization
typedef struct{
  //device configuration  
  Dim3 gridDim;
  Dim3 blockDim;
  unsigned numActiveBlocks;
  bool terminating;
/*MOD_MAN*/  //define parameters here for all kernels
    Node* g_graph_nodes;
    int* g_graph_edges;
    int* g_graph_mask;
    int* g_updating_graph_mask;
    int* g_graph_visited;
    int* g_cost;
    int* g_over;
    int no_of_nodes;
} PR_k1k2R;

//Defines the object passed to each pthread instance for k1k2R
typedef struct {
    unsigned warpid;
    unsigned blockgroupid;
    pthread_barrier_t *barrier;
    pthread_barrier_t *barrier_all;
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;


/*MOD_AUTO*/ //k1k2R specific parameter object
    PR_k1k2R *kp;//define pointer to PR_k1k2R
} P_PR_k1k2R;

/*MOD_AUTO*/
void *k1k2R_kernel(void *arg);


void *k1k2R_kernel(void *arg)
{
/*MOD_AUTO*/
    P_PR_k1k2R *p=(P_PR_k1k2R *)arg;    //restore pthead data pointer
    PR_k1k2R *kp=p->kp;   //restore k1k2R config and para pointer

    //recover kernel configuration
    Dim3 &gridDim = kp->gridDim;
    Dim3 &blockDim = kp->blockDim;

    //expand the configuration to get the 1-d layout
    unsigned numBlocksPerGrid = gridDim.x*gridDim.y*gridDim.z;
    unsigned numThreadsPerBlock = blockDim.x*blockDim.y*blockDim.z;
    //Set the init_mask based on expanded 1-d configuratioh,init_mask shouldn't be modified
    unsigned _i=16*p->warpid;
    vint _threadIdx = _mm512_set_epi32(_i+15,_i+14,_i+13,_i+12,_i+11,_i+10,_i+9,_i+8,_i+7,_i+6,_i+5,_i+4,_i+3,_i+2,_i+1,_i);
    vmask init_mask = _mm512_cmp_epi32_mask(_threadIdx, _mm512_set1_epi32(numThreadsPerBlock), _MM_CMPINT_LT); 

    /*MOD_MAN*/ //Activate corresponding threadIdx, all vector-lane variant variables
    vint threadIdxx= _mm512_mask_rem_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x)) ;
    vint threadIdxy= _mm512_mask_rem_epi32(vseti(0),init_mask,_mm512_div_epi32(_threadIdx, vseti(blockDim.x)), vseti(blockDim.y)) ;
    vint threadIdxz= _mm512_mask_div_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x*blockDim.y)) ;

    bool &terminating=kp->terminating;

while(1){//loop starts

//KERNEL1 Execution Loop
{    
    pthread_barrier_wait(p->barrier_all);
    if (terminating==true) return NULL;
    //MOD_MAN//restore KERNEL1's parameters,USE reference
    //float *&=...;
    Node* &g_graph_nodes = kp->g_graph_nodes;
    int* &g_graph_edges = kp->g_graph_edges;
    int* &g_graph_mask = kp->g_graph_mask;
    int* &g_updating_graph_mask = kp->g_updating_graph_mask;
    int* &g_graph_visited = kp->g_graph_visited;
    int* &g_cost = kp->g_cost;
    int &no_of_nodes = kp->no_of_nodes;
 
    for(unsigned _bidx=p->blockgroupid; _bidx<numBlocksPerGrid; _bidx += kp->numActiveBlocks){
        Dim3 blockIdx;
        blockIdx.x = _bidx%gridDim.x; 
        blockIdx.y = (_bidx/gridDim.x)%gridDim.y;
        blockIdx.z = _bidx/(gridDim.x*gridDim.y);
        //MOD_MAN: KERNEL1 BODY STARTS//
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

        //MOD_MAN: KERNEL1 BODY ENDS//
    }

    pthread_barrier_wait(p->barrier_all);
    
}//ends


//KERNEL2 Execution Loop
{
    pthread_barrier_wait(p->barrier_all);
    if (terminating==true) return NULL;

    //MOD_MAN//restore KERNEL1's parameters,USE reference 
    int* &g_graph_mask = kp->g_graph_mask;
    int* &g_updating_graph_mask = kp->g_updating_graph_mask;
    int* &g_graph_visited = kp->g_graph_visited;
    int* &g_over = kp->g_over;
    int &no_of_nodes = kp->no_of_nodes;

    for(unsigned _bidx=p->blockgroupid; _bidx<numBlocksPerGrid; _bidx += kp->numActiveBlocks){
        Dim3 blockIdx;
        blockIdx.x = _bidx%gridDim.x; 
        blockIdx.y = (_bidx/gridDim.x)%gridDim.y;
        blockIdx.z = _bidx/(gridDim.x*gridDim.y);
        //MOD_MAN: KERNEL2 BODY STARTS//
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



        //MOD_MAN: KERNEL2 BODY ENDS//
        }
    pthread_barrier_wait(p->barrier_all);

}//ends


}//loop ends


}//PTHREAD function ends


//////////////////////////////////////////////////////////////////////////////////////////////////


/*MOD_AUTO*/
void k1k2R(unsigned numPtsPerCore,float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old,
                            /*MOD_MAN: k1k2R Para Begin*/ 

        Node* d_graph_nodes,
    int* d_graph_edges,
    int* d_graph_mask,
    int* d_updating_graph_mask,
    int* d_graph_visited,
    int* d_cost,
    int no_of_nodes
                            /*MOD_MAN: k1k2R Para End*/)
{
//dim3 is not bit-wise copiable to phi, so define a simpler struct here
Dim3 dimGrid,dimBlock;
dimGrid.x=dimGrid_old.x; dimGrid.y=dimGrid_old.y; dimGrid.z=dimGrid_old.z;
dimBlock.x=dimBlock_old.x; dimBlock.y=dimBlock_old.y; dimBlock.z=dimBlock_old.z;
//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
    //Below calculates pthreads configuration
    unsigned AvaiLogicCore=sysconf(_SC_NPROCESSORS_ONLN)-2;//get # of availabe logical cores, avoid the last 4(spare the last phisical core for OS&I/O)
    set_self_affixed_to_spare_core(sysconf(_SC_NPROCESSORS_ONLN)-1);

    if(DEBUG) {
        printf("\nNum of Available Logical Processors on Phi: %d. \n",AvaiLogicCore);
        printf("\nXeon Phi Master Thread. Runing on Logical Core: %d.\n",sched_getcpu()); fflush(0);
    }
    //BELOW PART CALCULATES DEVICE UTILIZATION AND PTHREAD CONFIGURATION
    unsigned AvaiNumPts=AvaiLogicCore/2*numPtsPerCore;//get # of availabe pthreads
    unsigned numTdsPerBlock=dimBlock.x*dimBlock.y*dimBlock.z;
    unsigned PtsPerBlock=numTdsPerBlock/16; if(0!=numTdsPerBlock%16) PtsPerBlock++;//# of pthreads assigned to a block based on cuThreads
    unsigned maxConcBlocks=AvaiNumPts/PtsPerBlock*usage; //Max num of concurrently running cuBlocks 
    if(maxConcBlocks==0) maxConcBlocks=1;  

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
    pthread_barrier_t *barrier_all=(pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));//pthreads grouped into the same block will share a barrier

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_k1k2R *p=(P_PR_k1k2R *)malloc(num_of_pts_launch*sizeof(P_PR_k1k2R));//array of object passed to each pthread
    
/*MOD_AUTO*/ //k1k2R configaration data and parameters
    PR_k1k2R ker_parameters;
    ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock;//Kernel configuration<<<B,T>>>
    ker_parameters.numActiveBlocks=numActiveBlocks;//# of concurrent blocks
    ker_parameters.terminating=false;

    //init each barrier to sync PtsPerBlock pthreads, PtsPerBlock is the num of pthreads that make up a Cuda Block
    for(unsigned i=0;i<numActiveBlocks;++i){  //So, each barrier acts as a __synchthreads()
        pthread_barrier_init(barrier+i,NULL,PtsPerBlock);
    }
    //init barrier all, used to sync master threads and "num_of_pts_launch" working threads
    pthread_barrier_init(barrier_all,NULL,num_of_pts_launch+1); 

    cpu_set_t phi_set;//phi_set used to set pthread affinity
    unsigned j=0; unsigned core=0; unsigned gap=2-numPtsPerCore;

    for (unsigned i=0; i<num_of_pts_launch; i++) {
        //Init parameters for pthread/cudaBlock
        p[i].warpid=i%PtsPerBlock; //set pthread's warpid,recall that each warp here has 16 lanes        
        p[i].blockgroupid=i/PtsPerBlock;//set pthread's blockgroupid
        p[i].barrier=barrier+p[i].blockgroupid;//pthreads in a block group share a barrier
        p[i].barrier_all=barrier_all;
        p[i].kp=&ker_parameters;//pass kernel config and parameters

        /*//MOD_MAN: ONLY USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
        int *s_mem;
        if(p[i].warpid==0) {
          s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
        }
        p[i].s_mem=s_mem;
        */
    
        //create pthread with affinity "core"
        CPU_ZERO(&phi_set); CPU_SET(core++,&phi_set); //set pthread affinity
        pthread_attr_setaffinity_np(&pt_attr,sizeof(cpu_set_t), &phi_set);

        /*MOD_AUTO*/ //create pthreads for k1k2R       
        pthread_create(&threads[i], &pt_attr,k1k2R_kernel, (void *)(p+i));        //create with affinity

        if(++j == numPtsPerCore) {//This if clause implements affinity, each phisical core will run a fixed "numPtsPerCore" pthreads. 
            j=0; core+=gap;
        }
    }

//MOD_MAN: k1k2R CODE STARTS
    int stop;
    do{
        stop=0;

    //MOD_MAN//SET parameters for the first KERNEL
    //ker_parameters.arr=arr; 
    ker_parameters.g_graph_nodes = d_graph_nodes;
    ker_parameters.g_graph_edges = d_graph_edges;
    ker_parameters.g_graph_mask = d_graph_mask;
    ker_parameters.g_updating_graph_mask = d_updating_graph_mask;
    ker_parameters.g_graph_visited = d_graph_visited;
    ker_parameters.g_cost = d_cost;
    ker_parameters.no_of_nodes = no_of_nodes;
    
    pthread_barrier_wait(p->barrier_all);//start the first kernel   
    pthread_barrier_wait(p->barrier_all);//wait for the first kernel to finish
   


    
    //MOD_MAN//SET parameters for the second KERNEL
    //ker_parameters.arr=arr;
    ker_parameters.g_graph_mask = d_graph_mask;
    ker_parameters.g_updating_graph_mask = d_updating_graph_mask;
    ker_parameters.g_graph_visited = d_graph_visited;
    ker_parameters.g_over = &stop;
    ker_parameters.no_of_nodes = no_of_nodes; 

    pthread_barrier_wait(p->barrier_all);     //start the second kernel  
    pthread_barrier_wait(p->barrier_all);     //wait for the first kernel to finish
    

    
    }
    while(stop);


//k1k2R CODE ENDS
    ker_parameters.terminating=true;
    pthread_barrier_wait(p->barrier_all);
//

    //Destory barriers
    for(unsigned i=0;i<numActiveBlocks;++i){
        pthread_barrier_destroy(barrier+i);
    }
    pthread_barrier_destroy(barrier_all);
    //MOD_MAN: Free Shared Mem
    //Free data
    free(p);free(threads);free(barrier);


}






















#endif
