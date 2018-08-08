    //Below calculates pthreads configuration

#ifndef PROCESSOR
    unsigned AvaiLogicCore=sysconf(_SC_NPROCESSORS_ONLN)-4;//get # of availabe logical cores, avoid the last 4(spare the last phisical core for OS&I/O)
    set_self_affixed_to_spare_core(sysconf(_SC_NPROCESSORS_ONLN)-1);
#else
    unsigned AvaiLogicCore=sysconf(_SC_NPROCESSORS_ONLN);
    set_self_affixed_to_spare_core(sysconf(_SC_NPROCESSORS_ONLN)-1);
#endif

    if(DEBUG) {
        printf("\nNum of Available Logical Processors on Phi: %d. \n",AvaiLogicCore);
        printf("\nXeon Phi Master Thread. Runing on Logical Core: %d.\n",sched_getcpu()); fflush(0);
    }
    //BELOW PART CALCULATES DEVICE UTILIZATION AND PTHREAD CONFIGURATION
#ifndef SKYLAKE
    unsigned AvaiNumPts=AvaiLogicCore/4*numPtsPerCore;//get # of availabe pthreads
#else
    unsigned AvaiNumPts=AvaiLogicCore/2*numPtsPerCore;//get # of availabe pthreads
#endif
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
