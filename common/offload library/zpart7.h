//init each barrier to sync PtsPerBlock pthreads, PtsPerBlock is the num of pthreads that make up a Cuda Block
    for(unsigned i=0;i<numActiveBlocks;++i){  //So, each barrier acts as a __synchthreads()
        pthread_barrier_init(barrier+i,NULL,PtsPerBlock);
    }

    cpu_set_t phi_set;//phi_set used to set pthread affinity
#ifndef SKYLAKE
    unsigned j=0; unsigned core=0; unsigned gap=4-numPtsPerCore;
#else
    unsigned j=0; unsigned core=0; unsigned gap=2-numPtsPerCore;
#endif

    for (unsigned i=0; i<num_of_pts_launch; i++) {
        //Init parameters for pthread/cudaBlock
        p[i].warpid=i%PtsPerBlock; //set pthread's warpid,recall that each warp here has 16 lanes
