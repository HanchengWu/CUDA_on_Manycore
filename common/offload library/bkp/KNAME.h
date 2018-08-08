//Replace "KNAME" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __KNAME__
#define __KNAME__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
  //device configuration  
  Dim3 gridDim;
  Dim3 blockDim;
  unsigned numActiveBlocks;

/*MOD_MAN*/  //define kernel parameters here for Kernel KNAME
//int para;
} PR_KNAME;

//Defines the object passed to each pthread instance for KNAME
typedef struct {
    unsigned warpid;
    unsigned blockgroupid;
    pthread_barrier_t *barrier;
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_KNAME *kp;//define pointer to PR_KNAME
} P_PR_KNAME;

/*MOD_AUTO*/
void *KNAME(void *arg);

#pragma offload_attribute (pop)

void *KNAME(void *arg)
{
/*MOD_AUTO*/
    P_PR_KNAME *p=(P_PR_KNAME *)arg;    //restore pthead data pointer
    PR_KNAME *kp=p->kp;   //restore KNAME config and para pointer
    
/*MOD_MAN*///restore kernel KNAME's parameters
    //int para=kp->para;
//restore ends   

    //recover kernel configuration
    Dim3 gridDim = kp->gridDim;
    Dim3 blockDim = kp->blockDim;

    //expand the configuration to get the 1-d layout
    unsigned numBlocksPerGrid = gridDim.x*gridDim.y*gridDim.z;
    unsigned numThreadsPerBlock = blockDim.x*blockDim.y*blockDim.z;
    //Set the init_mask based on expanded 1-d configuratioh,init_mask shouldn't be modified
    unsigned _i=16*p->warpid;
    vint _threadIdx = _mm512_set_epi32(_i+15,_i+14,_i+13,_i+12,_i+11,_i+10,_i+9,_i+8,_i+7,_i+6,_i+5,_i+4,_i+3,_i+2,_i+1,_i);
    vmask init_mask = _mm512_cmp_epi32_mask(_threadIdx, _mm512_set1_epi32(numThreadsPerBlock), _MM_CMPINT_LT); 

    /*MOD_MAN*/ //Activate corresponding threadIdx, all vector-lane variant variables
    vint threadIdxx= _mm512_mask_rem_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x)) ;
    //vint ty_tmp= _mm512_div_epi32(_threadIdx, vseti(blockDim.x)) ;
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
        if(DEBUG) {
            int *tx=(int *)&threadIdxx;
            int *ty=(int *)&threadIdxy;
            int *tz=(int *)&threadIdxz;

            int init_mask_int = _mm512_mask2int(init_mask);
            int ls = 0;
            for ( int j = 0; j < 16; j++ ) {
                int tmp=init_mask_int&0x0001;
                if (tmp==0) {ls=j;break;}
                init_mask_int = init_mask_int>>1;
            }
            ls = ls==0?16:ls;
            printf("\nI am block(%u, %u, %u), warp:%u, barrier pointer:%X,threadIdx.x: %d ~ %d, threadIdx.y: %d ~ %d ,threadIdx.z: %d ~ %d, runing on Logical Core: %d. numBlocks/Grid:%d, numActiveBlocks:%d.\n", \
            blockIdx.x, blockIdx.y, blockIdx.z, p->warpid, (void *)p->barrier, tx[0], tx[ls-1], ty[0], ty[ls-1], tz[0], tz[ls-1], sched_getcpu(), numBlocksPerGrid, kp->numActiveBlocks); fflush(0);
        }
    }
    return (NULL);
}

/*MOD_AUTO*/
void KNAME_launch(unsigned numPtsPerCore, const dim3 &dimGrid_old, const dim3 &dimBlock_old,
                            /*MOD_MAN: Para Begin*/ 
                            /*MOD_MAN: Para End*/)
{
//dim3 is not bit-wise copiable to phi, so define a simpler struct here
Dim3 dimGrid,dimBlock;
dimGrid.x=dimGrid_old.x; dimGrid.y=dimGrid_old.y; dimGrid.z=dimGrid_old.z;
dimBlock.x=dimBlock_old.x; dimBlock.y=dimBlock_old.y; dimBlock.z=dimBlock_old.z;
//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
/*MOD_MAN Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN Para*/     //           in( num: ONCE  ) 
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

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_KNAME *p=(P_PR_KNAME *)malloc(num_of_pts_launch*sizeof(P_PR_KNAME));//array of object passed to each pthread
    
/*MOD_AUTO*/ //KNAME configaration data and parameters
    PR_KNAME ker_parameters;

    ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock;//Kernel configuration<<<B,T>>>
    ker_parameters.numActiveBlocks=numActiveBlocks;//# of concurrent blocks

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    //ker_parameters.arr=arr; 

    //init each barrier to sync PtsPerBlock pthreads, PtsPerBlock is the num of pthreads that make up a Cuda Block
    for(unsigned i=0;i<numActiveBlocks;++i){  //So, each barrier acts as a __synchthreads()
        pthread_barrier_init(barrier+i,NULL,PtsPerBlock);
    }

    cpu_set_t phi_set;//phi_set used to set pthread affinity
    unsigned j=0; unsigned core=0; unsigned gap=4-numPtsPerCore;

    for (unsigned i=0; i<num_of_pts_launch; i++) {
        //Init parameters for pthread/cudaBlock
        p[i].warpid=i%PtsPerBlock; //set pthread's warpid,recall that each warp here has 16 lanes

/*MOD_MAN*///ONLY USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
        //if(p[i].warpid==0) {
        //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
        //}
        //SHARED MEM ENDS
        
        p[i].blockgroupid=i/PtsPerBlock;//set pthread's blockgroupid
        p[i].barrier=barrier+p[i].blockgroupid;//pthreads in a block group share a barrier
        p[i].kp=&ker_parameters;//pass kernel config and parameters
    
        //create pthread with affinity "core"
        CPU_ZERO(&phi_set); CPU_SET(core++,&phi_set); //set pthread affinity
        pthread_attr_setaffinity_np(&pt_attr,sizeof(cpu_set_t), &phi_set);

/*MOD_AUTO*/ //create pthreads for KNAME       
        pthread_create(&threads[i], &pt_attr,KNAME, (void *)(p+i));        //create with affinity

        if(++j == numPtsPerCore) {//This if clause implements affinity, each phisical core will run a fixed "numPtsPerCore" pthreads. 
            j=0; core+=gap;
        }
    }

    // Synchronize to wait the completion of each thread. 
    for (int i=0; i<num_of_pts_launch; i++) {
        pthread_join(threads[i],NULL);
    }
    //Destory barriers
    for(unsigned i=0;i<numActiveBlocks;++i){
        pthread_barrier_destroy(barrier+i);
    }
    //Free data
    free(p);free(threads);free(barrier);
    }//offload ends

}

#endif
