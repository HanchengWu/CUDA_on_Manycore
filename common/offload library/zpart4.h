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