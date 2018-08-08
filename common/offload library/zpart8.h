p[i].blockgroupid=i/PtsPerBlock;//set pthread's blockgroupid
        p[i].barrier=barrier+p[i].blockgroupid;//pthreads in a block group share a barrier
        p[i].kp=&ker_parameters;//pass kernel config and parameters
    
        //create pthread with affinity "core"
        CPU_ZERO(&phi_set); CPU_SET(core++,&phi_set); //set pthread affinity
        pthread_attr_setaffinity_np(&pt_attr,sizeof(cpu_set_t), &phi_set);
