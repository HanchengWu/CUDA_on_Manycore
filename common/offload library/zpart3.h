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