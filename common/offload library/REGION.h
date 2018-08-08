//Replace "REGION" with your "REGION NAME", then modify places marked with "MOD_MAN"
#ifndef __REGION__
#define __REGION__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
  //device configuration  
  Dim3 gridDim;
  Dim3 blockDim;
  unsigned numActiveBlocks;
/*MOD_MAN*/  //define parameters here for all kernels





} PR_REGION;

//Defines the object passed to each pthread instance for REGION
typedef struct {
    unsigned warpid;
    unsigned blockgroupid;
    pthread_barrier_t *barrier;
    pthread_barrier_t *barrier_all;
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;


/*MOD_AUTO*/ //Region specific parameter object
    PR_REGION *kp;//define pointer to PR_REGION
} P_PR_REGION;

/*MOD_AUTO*/
void *REGION(void *arg);

#pragma offload_attribute (pop)

void *REGION(void *arg)
{
/*MOD_AUTO*/
    P_PR_REGION *p=(P_PR_REGION *)arg;    //restore pthead data pointer
    PR_REGION *kp=p->kp;   //restore REGION config and para pointer

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

    //KERNEL1 Execution Loop
    {    
    pthread_barrier_wait(p->barrier_all);
    //MOD_MAN//restore KERNEL1's parameters,USE reference
 
    for(unsigned _bidx=p->blockgroupid; _bidx<numBlocksPerGrid; _bidx += kp->numActiveBlocks){
        Dim3 blockIdx;
        blockIdx.x = _bidx%gridDim.x; 
        blockIdx.y = (_bidx/gridDim.x)%gridDim.y;
        blockIdx.z = _bidx/(gridDim.x*gridDim.y);
    //MOD_MAN: KERNEL1 BODY STARTS//



    //MOD_MAN: KERNEL1 BODY ENDS//
    }pthread_barrier_wait(p->barrier_all);
    
    }

/*
    //KERNEL2 Execution Loop
    {
    pthread_barrier_wait(p->barrier_all);
    
    //MOD_MAN//restore KERNEL1's parameters,USE reference 
    for(unsigned _bidx=p->blockgroupid; _bidx<numBlocksPerGrid; _bidx += kp->numActiveBlocks){
        Dim3 blockIdx;
        blockIdx.x = _bidx%gridDim.x; 
        blockIdx.y = (_bidx/gridDim.x)%gridDim.y;
        blockIdx.z = _bidx/(gridDim.x*gridDim.y);
    //MOD_MAN: KERNEL2 BODY STARTS//



    //MOD_MAN: KERNEL2 BODY ENDS//
    }pthread_barrier_wait(p->barrier_all);

    }
*/
    return NULL;
}


/*MOD_AUTO*/
void REGION_launch(unsigned numPtsPerCore, const dim3 &dimGrid_old, const dim3 &dimBlock_old
                            /*MOD_MAN: REGION Para Begin*/ 
                            /*MOD_MAN: REGION Para End*/)
{
//dim3 is not bit-wise copiable to phi, so define a simpler struct here
Dim3 dimGrid,dimBlock;
dimGrid.x=dimGrid_old.x; dimGrid.y=dimGrid_old.y; dimGrid.z=dimGrid_old.z;
dimBlock.x=dimBlock_old.x; dimBlock.y=dimBlock_old.y; dimBlock.z=dimBlock_old.z;
//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
/*MOD_MAN REGION Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN REGION Para*/     //           in( num: ONCE  ) 
    {//offload begins
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
    unsigned maxConcBlocks=AvaiNumPts/PtsPerBlock; //Max num of concurrently running cuBlocks 
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
    P_PR_REGION *p=(P_PR_REGION *)malloc(num_of_pts_launch*sizeof(P_PR_REGION));//array of object passed to each pthread
    
/*MOD_AUTO*/ //REGION configaration data and parameters
    PR_REGION ker_parameters;
    ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock;//Kernel configuration<<<B,T>>>
    ker_parameters.numActiveBlocks=numActiveBlocks;//# of concurrent blocks

    //init each barrier to sync PtsPerBlock pthreads, PtsPerBlock is the num of pthreads that make up a Cuda Block
    for(unsigned i=0;i<numActiveBlocks;++i){  //So, each barrier acts as a __synchthreads()
        pthread_barrier_init(barrier+i,NULL,PtsPerBlock);
    }
    //init barrier all, used to sync master threads and "num_of_pts_launch" working threads
    pthread_barrier_init(barrier_all,NULL,num_of_pts_launch+1); 

    cpu_set_t phi_set;//phi_set used to set pthread affinity
    unsigned j=0; unsigned core=0; unsigned gap=4-numPtsPerCore;

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

        /*MOD_AUTO*/ //create pthreads for REGION       
        pthread_create(&threads[i], &pt_attr,REGION, (void *)(p+i));        //create with affinity

        if(++j == numPtsPerCore) {//This if clause implements affinity, each phisical core will run a fixed "numPtsPerCore" pthreads. 
            j=0; core+=gap;
        }
    }

//MOD_MAN: REGION CODE STARTS



    //MOD_MAN//SET parameters for the first KERNEL
    //ker_parameters.arr=arr; 
    
    pthread_barrier_wait(p->barrier_all);//start the first kernel   
    pthread_barrier_wait(p->barrier_all);//wait for the first kernel to finish
   


    /*
    //MOD_MAN//SET parameters for the second KERNEL
    //ker_parameters.arr=arr; 

    pthread_barrier_wait(p->barrier_all);     //start the second kernel  
    pthread_barrier_wait(p->barrier_all);     //wait for the first kernel to finish
    */

    

//REGION CODE ENDS

    //Destory barriers
    for(unsigned i=0;i<numActiveBlocks;++i){
        pthread_barrier_destroy(barrier+i);
    }
    pthread_barrier_destroy(barrier_all);
    //MOD_MAN: Free Shared Mem
    //Free data
    free(p);free(threads);free(barrier);
    }//offload ends

}




#endif
