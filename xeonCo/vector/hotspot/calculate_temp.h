//Replace "calculate_temp" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __calculate_temp__
#define __calculate_temp__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
  #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel calculate_temp
  int iteration;
  float *power;
  float *temp_src;
  float *temp_dst;
  int grid_cols;
  int grid_rows;
  int border_cols;
  int border_rows;
  float Cap;
  float Rx;
  float Ry;
  float Rz;
  float step;
  float time_elapsed;
} PR_calculate_temp;

//Defines the object passed to each pthread instance for calculate_temp
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    float *temp_on_cuda;
    float *power_on_cuda;
    float *temp_t;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_calculate_temp *kp;//define pointer to PR_calculate_temp

} P_PR_calculate_temp;

/*MOD_AUTO*/
void *calculate_temp_KERNEL(void *arg);

inline vmask IN_RANGE_V(vint x, vint min, vint max, vmask oldm){
    vmask newm;
    //x>=min 
    newm = _mm512_mask_cmp_epi32_mask(oldm, x, min, _MM_CMPINT_GE);
    //&& x<=(max)
    newm = _mm512_mask_cmp_epi32_mask(newm, x, max, _MM_CMPINT_LE);
    return newm;
}


inline vint three_op(vint va, vint vb, int op, vint vt, vint vf, vmask oldm){

    vmask cm;
    switch (op){
        case _MM_CMPINT_EQ:
            cm = _mm512_cmp_epi32_mask(va, vb, _MM_CMPINT_EQ); break;
        case _MM_CMPINT_LT:
            cm = _mm512_cmp_epi32_mask(va, vb, _MM_CMPINT_LT); break;
        case _MM_CMPINT_LE:
            cm = _mm512_cmp_epi32_mask(va, vb, _MM_CMPINT_LE); break;
        case _MM_CMPINT_NE:
            cm = _mm512_cmp_epi32_mask(va, vb, _MM_CMPINT_NE); break;
        case _MM_CMPINT_NLT:
            cm = _mm512_cmp_epi32_mask(va, vb, _MM_CMPINT_NLT); break;    
        case _MM_CMPINT_NLE:
            cm = _mm512_cmp_epi32_mask(va, vb, _MM_CMPINT_NLE); break;
        default: 
            printf("\nOp code not supported\n"); fflush(0);
    }

    vint vv =_mm512_mask_mov_epi32(vf,cm,vt);
    return _mm512_mask_mov_epi32(vseti(0),oldm,vv);
}


#pragma offload_attribute (pop)


void *calculate_temp_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_calculate_temp *p=(P_PR_calculate_temp *)arg;    //restore pthead data pointer
    PR_calculate_temp *kp=p->kp;   //restore calculate_temp config and para pointer
    
    /*MOD_MAN*///restore kernel calculate_temp's parameters
    int iteration=kp->iteration;
    float *power=kp->power;
    float *temp_src=kp->temp_src;
    float *temp_dst=kp->temp_dst;
    int grid_cols=kp->grid_cols;
    int grid_rows=kp->grid_rows;
    int border_cols=kp->border_cols;
    int border_rows=kp->border_rows;
    float Cap=kp->Cap;
    float Rx=kp->Rx;
    float Ry=kp->Ry;
    float Rz=kp->Rz;
    float step=kp->step;
    float time_elapsed=kp->time_elapsed;

    /*MOD_MAN*///restore shared memory
    float *temp_on_cuda=p->temp_on_cuda;
    float *power_on_cuda=p->power_on_cuda;
    float *temp_t=p->temp_t;
   

    #include "zpart3.h"    //recover kernel configuration

    /*MOD_MAN*/ //Activate corresponding threadIdx, all vector-lane variant variables
    vint threadIdxx= _mm512_mask_rem_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x)) ;
    vint threadIdxy= _mm512_mask_rem_epi32(vseti(0),init_mask,_mm512_div_epi32(_threadIdx, vseti(blockDim.x)), vseti(blockDim.y)) ;
    vint threadIdxz= _mm512_mask_div_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x*blockDim.y)) ;

    vint izero=_mm512_set1_epi32(0); vfloat fzero=_mm512_set1_ps(0.0);

/*MOD_MAN*///Block invariant variables
    float amb_temp = 80.0;
    float step_div_Cap;
    float Rx_1,Ry_1,Rz_1;

    vint tx=threadIdxx;
    vint ty=threadIdxy;

    step_div_Cap=step/Cap;

    Rx_1=1/Rx;
    Ry_1=1/Ry;
    Rz_1=1/Rz;   

    int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
    int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

/*MOD_MAN*///Block invariant variables ends
    for(unsigned _bidx=p->blockgroupid; _bidx<numBlocksPerGrid; _bidx += kp->numActiveBlocks){
        /*MOD_MAN*/ //Activate corresponding blockIdx, all vector-lane invariant variables

        Dim3 blockIdx;
        blockIdx.x = _bidx%gridDim.x; 
        blockIdx.y = (_bidx/gridDim.x)%gridDim.y;
        blockIdx.z = _bidx/(gridDim.x*gridDim.y);
/*MOD_MAN*///KERNEL STARTS//

        int bx = blockIdx.x;
        int by = blockIdx.y;

        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+BLOCK_SIZE-1;
        int blkXmax = blkX+BLOCK_SIZE-1;

        //int yidx = blkY+ty;
        //int xidx = blkX+tx;
        vint yidx = _mm512_mask_add_epi32(izero,init_mask,vseti(blkY),ty);
        vint xidx = _mm512_mask_add_epi32(izero,init_mask,vseti(blkX),tx);

        //int index = grid_cols*loadYidx+loadXidx;
        vint index_p1=_mm512_mask_mullo_epi32(izero,init_mask,vseti(grid_cols), yidx);
        vint index = _mm512_mask_add_epi32(izero,init_mask,index_p1,xidx);


        //if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
        vmask mif1_t_p1 = IN_RANGE_V(yidx,izero,vseti(grid_rows-1),init_mask);
        vmask mif1_t = IN_RANGE_V(xidx,izero,vseti(grid_cols-1),mif1_t_p1);
            //temp_on_cuda[ty][tx] = temp_src[index];  
            vfloat tsi=_mm512_mask_i32gather_ps(fzero, mif1_t, index, temp_src, sizeof(float));//right
            vint toc_idx=_mm512_mask_mullo_epi32(izero,mif1_t,ty, vseti(BLOCK_SIZE));//left idx
            toc_idx=_mm512_mask_add_epi32(izero,mif1_t,toc_idx,tx);//left idx
            _mm512_mask_i32scatter_ps(temp_on_cuda, mif1_t, toc_idx, tsi, sizeof(float));
            
            //power_on_cuda[ty][tx] = power[index];
            vfloat pi=_mm512_mask_i32gather_ps(fzero, mif1_t, index, power, sizeof(float));//right
            _mm512_mask_i32scatter_ps(power_on_cuda, mif1_t, toc_idx, pi, sizeof(float));
        //}if scope ends

        //__syncthreads();
        pthread_barrier_wait(p->barrier);

        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

        //int N = ty-1;
        vint N=_mm512_mask_sub_epi32(izero,init_mask,ty,vseti(1));
        //int S = ty+1;
        vint S=_mm512_mask_add_epi32(izero,init_mask,ty,vseti(1));
        //int W = tx-1;
        vint W=_mm512_mask_sub_epi32(izero,init_mask,tx,vseti(1));
        //int E = tx+1;
        vint E=_mm512_mask_add_epi32(izero,init_mask,tx,vseti(1));

        //N = (N < validYmin) ? validYmin : N;
        N=three_op(N, vseti(validYmin), _MM_CMPINT_LT, vseti(validYmin), N, init_mask);
        //S = (S > validYmax) ? validYmax : S;
        S=three_op(S, vseti(validYmax), _MM_CMPINT_GT, vseti(validYmax), S, init_mask);
        //W = (W < validXmin) ? validXmin : W;
        W=three_op(W, vseti(validXmin), _MM_CMPINT_LT, vseti(validXmin), W, init_mask);
        //E = (E > validXmax) ? validXmax : E;
        E=three_op(E, vseti(validXmax), _MM_CMPINT_GT, vseti(validXmax), E, init_mask);

        //bool computed;
        vint computed;
  
        //compute index[tx][ty]
        vint tytx=_mm512_mask_mullo_epi32(izero,init_mask,ty, vseti(BLOCK_SIZE));//left idx
        tytx=_mm512_mask_add_epi32(izero,init_mask,tytx,tx);//left idx
        //compute index[S][tx]
        vint Stx=_mm512_mask_mullo_epi32(izero,init_mask,S, vseti(BLOCK_SIZE));//left idx
        Stx=_mm512_mask_add_epi32(izero,init_mask,Stx,tx);//left idx
        //compute index[S][tx]
        vint Ntx=_mm512_mask_mullo_epi32(izero,init_mask,N, vseti(BLOCK_SIZE));//left idx
        Ntx=_mm512_mask_add_epi32(izero,init_mask,Ntx,tx);//left idx
        //compute index[ty][E]
        vint tyE=_mm512_mask_mullo_epi32(izero,init_mask,ty, vseti(BLOCK_SIZE));//left idx
        tyE=_mm512_mask_add_epi32(izero,init_mask,tyE,E);//left idx
        //compute index[ty][W]
        vint tyW=_mm512_mask_mullo_epi32(izero,init_mask,ty, vseti(BLOCK_SIZE));//left idx
        tyW=_mm512_mask_add_epi32(izero,init_mask,tyW,W);//left idx

        for(int i=0; i<iteration; i++){
            //computed = false;
            computed = vseti(0);

            //if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  IN_RANGE(tx, validXmin, validXmax) && IN_RANGE(ty, validYmin, validYmax) ) {
            vmask mif2_t_p1=IN_RANGE_V(tx,vseti(i+1),vseti(BLOCK_SIZE-i-2),init_mask);
            vmask mif2_t_p2=IN_RANGE_V(ty,vseti(i+1),vseti(BLOCK_SIZE-i-2),mif2_t_p1);
            vmask mif2_t_p3=IN_RANGE_V(tx,vseti(validXmin),vseti(validXmax),mif2_t_p2);
            vmask mif2_t   =IN_RANGE_V(ty,vseti(validYmin),vseti(validYmax),mif2_t_p3);
                    computed=_mm512_mask_mov_epi32(computed,mif2_t,vseti(1));
            //      temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 
            //                         2.0*temp_on_cuda[ty][tx]) * Ry_1 + (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 
            //                        + (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
                    vfloat m0=_mm512_mask_i32gather_ps(fzero,mif2_t,tytx,temp_on_cuda,sizeof(float));
                    vfloat m1=_mm512_mask_i32gather_ps(fzero,mif2_t,tytx,power_on_cuda,sizeof(float));
                    vfloat m2=_mm512_mask_i32gather_ps(fzero,mif2_t,Stx,temp_on_cuda,sizeof(float));
                    vfloat m3=_mm512_mask_i32gather_ps(fzero,mif2_t,Ntx,temp_on_cuda,sizeof(float));
                    vfloat m5=_mm512_mask_i32gather_ps(fzero,mif2_t,tyE,temp_on_cuda,sizeof(float));
                    vfloat m6=_mm512_mask_i32gather_ps(fzero,mif2_t,tyW,temp_on_cuda,sizeof(float));
                    vfloat m0_2=_mm512_mask_mul_ps(fzero,mif2_t,vsetf(2.0),m0);
                    vfloat m7=_mm512_mask_add_ps(fzero,mif2_t,m2,m3);
                    m7=_mm512_mask_sub_ps(fzero,mif2_t,m7,m0_2);
                    vfloat m8=_mm512_mask_add_ps(fzero,mif2_t,m5,m6);
                    m8=_mm512_mask_sub_ps(fzero,mif2_t,m8,m0_2);
                    vfloat m9 = _mm512_mask_sub_ps(fzero,mif2_t,vsetf(amb_temp),m0);
                    vfloat m10 = _mm512_mask_mul_ps(fzero,mif2_t,m7,vsetf(Ry_1)) ;
                    vfloat m11 = _mm512_mask_mul_ps(fzero,mif2_t,m8,vsetf(Rx_1)) ;
                    vfloat m12 = _mm512_mask_mul_ps(fzero,mif2_t,m9,vsetf(Rz_1)) ;
                    vfloat m13 = _mm512_mask_add_ps(fzero,mif2_t,m1,m10);
                    m13 = _mm512_mask_add_ps(fzero,mif2_t,m13,m11);
                    m13 = _mm512_mask_add_ps(fzero,mif2_t,m13,m12);
//                    vfloat m14 = m0 + step_div_Cap * m13;
                    vfloat m14 = _mm512_mask_fmadd_ps(vsetf(step_div_Cap),mif2_t,m13,m0);
                    //final commit
                    _mm512_mask_i32scatter_ps(temp_t, mif2_t, tytx, m14, sizeof(float));
            //}if ends

            //__syncthreads();
            pthread_barrier_wait(p->barrier);

            if(i==iteration-1) break;
            
            //if(computed) 
            vmask mif4_t = _mm512_mask_cmp_epi32_mask(init_mask, computed, vseti(1), _MM_CMPINT_EQ);
                //temp_on_cuda[ty][tx]= temp_t[ty][tx];
                vfloat tttx2 = _mm512_mask_i32gather_ps(fzero, mif4_t, tytx, temp_t, sizeof(float));//right
                _mm512_mask_i32scatter_ps(temp_on_cuda, mif4_t, tytx, tttx2, sizeof(float));
            //if ends
            
            //__syncthreads();
            pthread_barrier_wait(p->barrier);
        }


        //if (computed){
        vmask mif3_t = _mm512_mask_cmp_epi32_mask(init_mask, computed, vseti(1), _MM_CMPINT_EQ);
            //temp_dst[index]= temp_t[ty][tx];
            vfloat tttx = _mm512_mask_i32gather_ps(fzero, mif3_t, tytx, temp_t, sizeof(float));//right
            _mm512_mask_i32scatter_ps(temp_dst, mif3_t, index, tttx, sizeof(float));
        //}//if loop ends

/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void calculate_temp(unsigned numPtsPerCore, float usage,const dim3 &dimGrid_old, const dim3 &dimBlock_old,
                    /*MOD_MAN: Para Begin*/ 
                    int iteration,  //number of iteration
                    float *power,   //power input
                    float *temp_src,    //temperature input/output
                    float *temp_dst,    //temperature input/output
                    int grid_cols,  //Col of grid
                    int grid_rows,  //Row of grid
                    int border_cols,  // border offset 
                    int border_rows,  // border offset
                    float Cap,      //Capacitance
                    float Rx, 
                    float Ry, 
                    float Rz, 
                    float step, 
                    float time_elapsed
                    /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(usage: ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
/*MOD_MAN Para*/                in( iteration: ONCE   ) \
/*MOD_MAN Para*/                in( power: length(0) REUSE_A  ) \
                                in( temp_src: length(0) REUSE_A) \
                                in( temp_dst: length(0) REUSE_A) \
                                in( grid_cols: ONCE) \
                                in( grid_rows: ONCE) \
                                in( border_cols: ONCE) \
                                in( border_rows: ONCE) \
                                in( Cap: ONCE) \
                                in( Rx: ONCE) \
                                in( Ry: ONCE) \
                                in( Rz: ONCE) \
                                in( step: ONCE) \
                                in( time_elapsed: ONCE)    
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_calculate_temp *p=(P_PR_calculate_temp *)malloc(num_of_pts_launch*sizeof(P_PR_calculate_temp));//array of object passed to each pthread
    
/*MOD_AUTO*/ //calculate_temp configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_calculate_temp ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    ker_parameters.iteration=iteration ;
    ker_parameters.power=power ;
    ker_parameters.temp_src=temp_src ;
    ker_parameters.temp_dst=temp_dst ;
    ker_parameters.grid_cols=grid_cols ;
    ker_parameters.grid_rows=grid_rows ;
    ker_parameters.border_cols=border_cols ;
    ker_parameters.border_rows=border_rows ;
    ker_parameters.Cap=Cap ;
    ker_parameters.Rx=Rx ;
    ker_parameters.Ry=Ry ;
    ker_parameters.Rz=Rz ;
    ker_parameters.step=step ;
    ker_parameters.time_elapsed ;

    float *temp_on_cuda,*power_on_cuda,*temp_t;
    #include "zpart7.h"

    /*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    if(p[i].warpid==0) {
        temp_on_cuda=(float *)malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(float));
        power_on_cuda=(float *)malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(float));
        temp_t=(float *)malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(float));
    }
    p[i].temp_on_cuda=temp_on_cuda;
    p[i].power_on_cuda=power_on_cuda;
    p[i].temp_t=temp_t;
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for calculate_temp       
    pthread_create(&threads[i], &pt_attr,calculate_temp_KERNEL, (void *)(p+i));        //create with affinity

    #include "zpart9.h"

    /*MOD_MAN*///free shared memory
    for (int i=0; i<num_of_pts_launch; i++) {
        if(p[i].warpid==0){
            free(p[i].temp_on_cuda); 
            free(p[i].power_on_cuda); 
            free(p[i].temp_t); 
        }
    } 
    /*MOD_MAN*///Ends

    //Free data
    free(p);free(threads);free(barrier);
    }//Offload ends

}
#endif



            //if(0==blockIdx.x && 0==blockIdx.y) {
            //    printf("\nAAA: block(%u,%u,%u), Warpid: %u, Pointer:%u.\n",blockIdx.x,blockIdx.y,blockIdx.z, p->warpid,(void *)temp_on_cuda); fflush(0);
            //    for (int m;m<16*16;m++){
            //        temp_on_cuda[m]=1.0;
            //    }
            //}
            //_mm512_mask_i32scatter_ps(temp_dst, mif1_t, toc_idx, tsi, sizeof(float));

        //FAKE
        //computed = vseti(1);
        //for (int m;m<16*16;m++){
        //    temp_t[m]=1.0;
        //}

/*
vfloat m0 = temp_on_cuda[ty][tx];
vfloat m1 = power_on_cuda[ty][tx];

vfloat m2 = temp_on_cuda[S][tx];
vfloat m3 = temp_on_cuda[N][tx];
      
vfloat m5 = temp_on_cuda[ty][E];
vfloat m6 = temp_on_cuda[ty][W];

vfloat m0_2=2.0*m0;

vfloat m7 = m2+ m3 - m0_2;
vfloat m8 = m5 + m6 - m0_2;
vfloat m9 = amb_temp - m0;

vfloat m10 = m7*Ry_1;
vfloat m11 = m8*Rx_1;
vfloat m12 = m9*Rz_1;
vfloat m13 = m1+ m10 + m11 + m12;
vfloat m14 = m0 + step_div_Cap * m13;

temp_t[ty][tx] =  m14;
*/
