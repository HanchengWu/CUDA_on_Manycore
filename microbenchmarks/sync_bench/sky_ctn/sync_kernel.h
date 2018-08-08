//Replace "sync_kernel" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __sync_kernel__
#define __sync_kernel__

//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel sync_kernel
    int *arr;
    int N;
    int *rdom_arr;
    int *result;

} PR_sync_kernel;

//Defines the object passed to each pthread instance for sync_kernel
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_sync_kernel *kp;//define pointer to PR_sync_kernel

} P_PR_sync_kernel;

/*MOD_AUTO*/
void *sync_kernel_KERNEL(void *arg);


void *sync_kernel_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_sync_kernel *p=(P_PR_sync_kernel *)arg;    //restore pthead data pointer
    PR_sync_kernel *kp=p->kp;   //restore sync_kernel config and para pointer
    
    /*MOD_MAN*///restore kernel sync_kernel's parameters
    int *arr=kp->arr;
    int N=kp->N;
    int *rdom_arr=kp->rdom_arr;
    int *result=kp->result;

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

        vint tem=vseti(0);
        vint random;
        vint tid= _mm512_mask_add_epi32(vseti(0),init_mask,threadIdxx,vseti(blockIdx.x*blockDim.x));
        int totalthreads=blockDim.x*gridDim.x;
    
    for (int i=0;i<10;++i){

        vint j=tid;
        vmask mk=_mm512_mask_cmp_epi32_mask(init_mask,j,vseti(N),_MM_CMPINT_LT);
        while(mk!=0){
        random = _mm512_mask_i32gather_epi32(vseti(0),mk,j,rdom_arr,sizeof(int));
        vint idx= _mm512_mask_rem_epi32(vseti(0),mk,  _mm512_mask_add_epi32(vseti(0),mk,j,random),vseti(N) );

        vint lvalue=_mm512_mask_i32gather_epi32(vseti(0),mk,idx,arr,sizeof(int));
        //vint lvalue=_mm512_mask_i32gather_epi32(vseti(0),mk,vseti(0),arr,sizeof(int));
        tem=_mm512_mask_add_epi32(vseti(0),mk,tem,lvalue);

        //sync here
        //pthread_barrier_wait(p->barrier);

        _mm512_mask_i32scatter_epi32(result,init_mask,tid,tem,sizeof(int));
        //_mm512_mask_i32scatter_epi32(result,init_mask,vseti(0),tem,sizeof(int));

        j=_mm512_mask_add_epi32(j,init_mask,j,vseti(totalthreads));
        mk=_mm512_mask_cmp_epi32_mask(mk,j,vseti(N),_MM_CMPINT_LT );
        }
    
    }


/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void sync_kernel(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old, 
int *arr, int N, int *rdom_arr, int *result /*MOD_MAN: Para Begin*/ /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
    //Below calculates pthreads configuration
        //Below calculates pthreads configuration

#ifndef PROCESSOR
    unsigned AvaiLogicCore=sysconf(_SC_NPROCESSORS_ONLN)-4;//get # of availabe logical cores, avoid the last 4(spare the last phisical core for OS&I/O)
    set_self_affixed_to_spare_core(sysconf(_SC_NPROCESSORS_ONLN)-1);
#else
    unsigned AvaiLogicCore=sysconf(_SC_NPROCESSORS_ONLN);
    printf("AvaiLogicCore: %d\n", AvaiLogicCore);
    set_self_affixed_to_spare_core(sysconf(_SC_NPROCESSORS_ONLN)-1);
#endif


    if(DEBUG) {
        printf("\nNum of Available Logical Processors on Phi: %d. \n",AvaiLogicCore);
        printf("\nXeon Phi Master Thread. Runing on Logical Core: %d.\n",sched_getcpu()); fflush(0);
    }
    //BELOW PART CALCULATES DEVICE UTILIZATION AND PTHREAD CONFIGURATION
    unsigned AvaiNumPts=AvaiLogicCore/4*numPtsPerCore;//get # of availabe pthreads
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


/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_sync_kernel *p=(P_PR_sync_kernel *)malloc(num_of_pts_launch*sizeof(P_PR_sync_kernel));//array of object passed to each pthread
    
/*MOD_AUTO*/ //sync_kernel configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_sync_kernel ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    ker_parameters.arr=arr;
    ker_parameters.N=N;
    ker_parameters.rdom_arr=rdom_arr;
    ker_parameters.result=result; 

//init each barrier to sync PtsPerBlock pthreads, PtsPerBlock is the num of pthreads that make up a Cuda Block
    for(unsigned i=0;i<numActiveBlocks;++i){  //So, each barrier acts as a __synchthreads()
        pthread_barrier_init(barrier+i,NULL,PtsPerBlock);
    }

    cpu_set_t phi_set;//phi_set used to set pthread affinity
    unsigned j=0; unsigned core=0; unsigned gap=2-numPtsPerCore;

    for (unsigned i=0; i<num_of_pts_launch; i++) {
        //Init parameters for pthread/cudaBlock
        p[i].warpid=i%PtsPerBlock; //set pthread's warpid,recall that each warp here has 16 lanes        
        p[i].blockgroupid=i/PtsPerBlock;//set pthread's blockgroupid
        p[i].barrier=barrier+p[i].blockgroupid;//pthreads in a block group share a barrier
        p[i].kp=&ker_parameters;//pass kernel config and parameters
    
        //create pthread with affinity "core"
        CPU_ZERO(&phi_set); CPU_SET(core++,&phi_set); //set pthread affinity
        pthread_attr_setaffinity_np(&pt_attr,sizeof(cpu_set_t), &phi_set);

/*MOD_AUTO*/ //create pthreads for sync_kernel       
    pthread_create(&threads[i], &pt_attr,sync_kernel_KERNEL, (void *)(p+i));        //create with affinity

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

    /*MOD_MAN*///free shared memory

    /*MOD_MAN*///Ends

    //Free data
    free(p);free(threads);free(barrier);


}
#endif
