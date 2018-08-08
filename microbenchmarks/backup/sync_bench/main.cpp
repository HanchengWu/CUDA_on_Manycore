#include "phi_template.h"

#define DEBUG 1
#define VERIFY 0

unsigned DEV_NUM = 1;

int DefaultNumBlocks=56*2;//This is defaulted num of XeonphiPthreads(aka.cuda blocks)
int DefaultNumThreads=16;
int DefaultnumPtsPerCore=2;
#pragma offload_attribute (push, target(mic))

//set up these config value before every kernel launch/offloading
unsigned numCuBlocks;
unsigned numCuThreads;
unsigned numPtsPerCore;

inline void set_self_affixed_to_spare_core(unsigned sparecore){
        cpu_set_t phi_set;//phi_set used to set pthread affinity
        CPU_ZERO(&phi_set); CPU_SET(sparecore,&phi_set);
        pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t), &phi_set);
}
//INCLUDE HEADERS TO OFFLOADED TO PHI HERE
int *arr;
int num;

//Struct holds kernel config and info used for synchronization
typedef struct{
  //device configuration  
  unsigned numCuBlocks;
  unsigned numCuThreads;
  unsigned numActiveBlocks;
  //define kernel parameters for KERNEL 1
  int *arr;
  int num;
} PR_KERNEL1;


//Defines the object passed to each pthread instance
typedef struct {
    unsigned warpid;
    unsigned blockgroupid;
    pthread_barrier_t *barrier;
    //shared mem pointer
    int *s_mem;

    PR_KERNEL1 *kp;//define pointer to PR_KERNEL1
} P_PR_KERNEL1;

typedef struct {
    unsigned pid;
    pthread_t pd;
    unsigned depth;
    pthread_barrier_t *bar_all;
    pthread_barrier_t *bar_block;
} P_DATA;

void *KERNEL1(void *arg)
{
    P_DATA *pp=(P_DATA *)arg;
    int depth=pp->depth;
    printf("\nI am at depth:%d, core:%d.\n",depth,sched_getcpu()); fflush(0);
    pthread_t pd[2];
    P_DATA pdata[2];
    //pthread_create(&new_pd, NULL,KERNEL1, (void *)(long)(depth-1));        //create with affinity
    if(depth>1){
        for (int i=0; i<2; i++) {
            //Init parameters for pthread/cudaBlock
            pdata[i].depth=depth-1;  
            pdata[i].bar_all=pp->bar_all;          
            pthread_create(&pd[i], NULL,KERNEL1, (void *)(pdata+i));        //create with affinity
        }
    }

    pthread_barrier_wait(pp->bar_all);
    //sleep(2);
    pthread_exit(NULL);
}
#pragma offload_attribute (pop)


int main(int argc, char *argv[]){
	if (4!= argc) {
		printf("Arguments is less then enough!\n");
		exit(0);
	}

	double time;
    DefaultNumBlocks=atoi(argv[2]);//Can change default # of Blocks and Threads through terminal
    DefaultNumThreads=atoi(argv[3]);
    DefaultnumPtsPerCore=atoi(argv[1]);

    unsigned AvaiLogicCore;
    
	#pragma offload target(mic: DEV_NUM) out(AvaiLogicCore: alloc_if(1) free_if(1)) 	
	{
		printf("\nInitialize PHI\n");
		fflush(0);//flush the buffer on Xeon Phi in time
        AvaiLogicCore=sysconf(_SC_NPROCESSORS_ONLN)-4;//get # of availabe logical cores, avoid the last 4(spare the last phisical core for OS&I/O)
	}

    num=20;
	arr=(int *)malloc(num*sizeof(int));
	
	for(int i=0;i<num;++i){
                arr[i]=i;
	}

//Execution on Phi begins
//KERNEL configuration or change here
    numCuBlocks = DefaultNumBlocks; 
    numCuThreads = DefaultNumThreads;
    numPtsPerCore = DefaultnumPtsPerCore;
    printf("#ofCuda Blocks):%d, #ofCuda Threads:%d.\n", numCuBlocks, numCuThreads);
	
    printf("\nExecution on Xeon Phi begins!\n");
	time = gettime_ms();

    //unsigned AvaiLogicCore=sysconf(_SC_NPROCESSORS_ONLN)-4;//get # of availabe logical cores, avoid the last 4(spare the last phisical core for OS&I/O)
    unsigned AvaiNumPts=AvaiLogicCore/4*numPtsPerCore;//get # of availabe pthreads
    unsigned PtsPerBlock=numCuThreads/16; if(0!=numCuThreads%16) PtsPerBlock++;//# of pthreads assigned to a block based on cuThreads
    unsigned maxConcBlocks=AvaiNumPts/PtsPerBlock; //Max num of concurrently running cuBlocks 
    unsigned num_of_pts_launch,numActiveBlocks;

    if(maxConcBlocks>=numCuBlocks){ //launch smaller between maxConcBlocks and numCuBlocks
        num_of_pts_launch=numCuBlocks*PtsPerBlock;
        numActiveBlocks=numCuBlocks;
    }else{
        //This indicates pthreads grouped in blocks have to iteration to finish numCuBlocks
        num_of_pts_launch=maxConcBlocks*PtsPerBlock;
        numActiveBlocks=maxConcBlocks;
    }

    if(0==numActiveBlocks) { 
        printf("\nBlock is configured too large to be put on Xeon phi.\n");
    }
	
	#pragma offload target(mic: DEV_NUM) 
	{//offload begins
    //Below calculates pthreads configuration
    set_self_affixed_to_spare_core(sysconf(_SC_NPROCESSORS_ONLN)-1);

    if(DEBUG) {
        printf("\nNum of Available Logical Processors on Phi: %d. \n",AvaiLogicCore);
        printf("\nXeon Phi Master Thread. Runing on Logical Core: %d.\n",sched_getcpu()); fflush(0);
        printf("\n# threads launched: %d.\n",num_of_pts_launch);fflush(0);

    }
    //BELOW PART CALCULATES DEVICE UTILIZATION AND PTHREAD CONFIGURATION
    
    //Create data hole pthreads ids, barriers and attibute
    pthread_t *threads=(pthread_t *)malloc(num_of_pts_launch*sizeof(pthread_t));//pthread id array
    pthread_attr_t pt_attr; pthread_attr_init(&pt_attr);//init pthread attribute
    pthread_barrier_t *barrier=(pthread_barrier_t *)malloc(numActiveBlocks*sizeof(pthread_barrier_t));//pthreads grouped into the same block will share a barrier
    //array of objects passed to each pthread
    P_PR_KERNEL1 *p=(P_PR_KERNEL1 *)malloc(num_of_pts_launch*sizeof(P_PR_KERNEL1));//array of object passed to each pthread

    //KERNEL1 configaration data and parameters
    PR_KERNEL1 ker_parameters;
    ker_parameters.numCuBlocks=numCuBlocks; ker_parameters.numCuThreads=numCuThreads;//Kernel configuration<<<B,T>>>
    ker_parameters.numActiveBlocks=numActiveBlocks;//# of concurrent blocks

    //init each barrier to sync PtsPerBlock pthreads, PtsPerBlock is the num of pthreads that make up a Cuda Block
    for(unsigned i=0;i<numActiveBlocks;++i){  //So, each barrier acts as a __synchthreads()
        pthread_barrier_init(barrier+i,NULL,PtsPerBlock);
    }

    pthread_barrier_t *barrier_all=(pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));//pthreads grouped into the same block will share a barrier
    pthread_barrier_init(barrier_all,NULL,31);


    cpu_set_t phi_set;//phi_set used to set pthread affinity
    unsigned j=0; unsigned core=0; unsigned gap=4-numPtsPerCore;
    //temporary shared mem pointers
    int * t_s_mem;
    P_DATA pdata[2];
    int depth=5;
    for (unsigned i=0; i<2; i++) {
        //Init parameters for pthread/cudaBlock
        p[i].warpid=i%PtsPerBlock; //set pthread's warpid,recall that each warp here has 16 lanes
        //ONLY USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
        if(p[i].warpid==0) {
          t_s_mem=(int *)malloc(sizeof(int)*numCuThreads);
        }
        p[i].s_mem=t_s_mem;
        //
        p[i].blockgroupid=i/PtsPerBlock;//set pthread's blockgroupid
        p[i].barrier=barrier+p[i].blockgroupid;//pthreads in a block group share a barrier
        p[i].kp=&ker_parameters;//pass kernel config and parameters
    
        //create pthread with affinity "core"
        CPU_ZERO(&phi_set); CPU_SET(core++,&phi_set); //set pthread affinity
        pthread_attr_setaffinity_np(&pt_attr,sizeof(cpu_set_t), &phi_set);
        pdata[i].depth=depth-1;
        pdata[i].bar_all=barrier_all;
        pthread_create(&threads[i], &pt_attr,KERNEL1, (void *)(pdata+i) );        //create with affinity
        printf("\nIterate\n");fflush(0);


        if(++j == numPtsPerCore) {//This if clause implements affinity, each phisical core will run a fixed "numPtsPerCore" pthreads. 
            j=0; core+=gap;
        }
    }

    printf("\nBBBBBB\n");fflush(0);
    pthread_barrier_wait(barrier_all);

    // Synchronize to wait the completion of each thread. 
    //for (int i=0; i<num_of_pts_launch; i++) {
    //    pthread_join(threads[i],NULL);
   //}
    //Destory barriers
    for(unsigned i=0;i<numActiveBlocks;++i){
        pthread_barrier_destroy(barrier+i);
    }
    printf("\nCCCC\n");fflush(0);
    //sleep(5);
    //Free data
    //free(p);free(threads);free(barrier);

	}//offload ends

    double time2=gettime_ms()-time;

    FILE *result=fopen("result.txt","a");
    fprintf(result,"\nDefaultnumPtsPerCore, %d, DefaultNumBlocks, %d, DefaultNumThreads, %d, ExecutionTime, %.2f \n", DefaultnumPtsPerCore, DefaultNumBlocks, DefaultNumThreads, time2);
	printf("\nExecution on Phi returns with time = %.2f ms.\n",time2);
    fclose(result);
	
	return 0;
}
