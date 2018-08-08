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

    
