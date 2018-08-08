//Replace "Fan2" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __Fan2__
#define __Fan2__

//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel Fan2
    float *m_cuda;
    float *a_cuda;
    float *b_cuda;
    int Size;
    int j1;
    int t;

} PR_Fan2;

//Defines the object passed to each pthread instance for Fan2
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_Fan2 *kp;//define pointer to PR_Fan2

} P_PR_Fan2;

/*MOD_AUTO*/
void *Fan2_KERNEL(void *arg);


void *Fan2_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_Fan2 *p=(P_PR_Fan2 *)arg;    //restore pthead data pointer
    PR_Fan2 *kp=p->kp;   //restore Fan2 config and para pointer
    
    /*MOD_MAN*///restore kernel Fan2's parameters
    float *m_cuda=kp->m_cuda;
    float *a_cuda=kp->a_cuda;
    float *b_cuda=kp->b_cuda;
    int Size=kp->Size;
    int j1=kp->j1;
    int t=kp->t;
    //int para=kp->para;
   

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

    //if( (threadIdx.x + blockIdx.x * blockDim.x < Size-1-t) && (threadIdx.y + blockIdx.y * blockDim.y < Size-t) ){
        vint gidx1= _mm512_mask_add_epi32(vseti(0),init_mask,vseti(blockDim.x*blockIdx.x),threadIdxx);
        vmask mif_t1=_mm512_mask_cmp_epi32_mask(init_mask, gidx1, vseti(Size-1-t), _MM_CMPINT_LT);
        vint gidx2= _mm512_mask_add_epi32(vseti(0),init_mask,vseti(blockIdx.y*blockDim.y),threadIdxy);
        vmask mif_t=_mm512_mask_cmp_epi32_mask(mif_t1, gidx2, vseti(Size-t), _MM_CMPINT_LT);
            vint xidx= gidx1;
            vint yidx= gidx2;

        //a_cuda[Size*(xidx+1+t)+(yidx+t)] =a_cuda[Size*(xidx+1+t)+(yidx+t)] - m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
            vint xidx1t=_mm512_mask_add_epi32(xidx,mif_t,xidx,vseti(1+t));
            //calculate p1=m_cuda[Size*xidx1t+t]
            vint midx=_mm512_mask_mullo_epi32(midx,mif_t,vseti(Size),xidx1t);
            midx=_mm512_mask_add_epi32(midx,mif_t,midx,vseti(t));
            vfloat p1=_mm512_mask_i32gather_ps(p1,mif_t,midx,m_cuda,sizeof(float));
            //calculate p2=a_cuda[Size*t+(yidx+t)]
            vint idxa=_mm512_mask_mullo_epi32(idxa,mif_t,vseti(Size),vseti(t));
            idxa=_mm512_mask_add_epi32(idxa,mif_t,idxa,yidx);
            idxa=_mm512_mask_add_epi32(idxa,mif_t,idxa,vseti(t));
            vfloat p2=_mm512_mask_i32gather_ps(p1,mif_t,idxa,a_cuda,sizeof(float));
            //calculate p3=p1*p2
            vfloat p3=_mm512_mask_mul_ps(vsetf(0),mif_t,p1,p2); 

            //calcualte a_cuda's first two common index
            vint aidx=_mm512_mask_mullo_epi32(aidx,mif_t,vseti(Size),xidx1t);
            aidx=_mm512_mask_add_epi32(aidx,mif_t,aidx,yidx);
            aidx=_mm512_mask_add_epi32(aidx,mif_t,aidx,vseti(t));
            //read a_cuda[Size*(xidx+1+t)+(yidx+t)]
            vfloat acdvalue=_mm512_mask_i32gather_ps(acdvalue,mif_t,aidx,a_cuda,sizeof(float));
            //a_cuda[Size*(xidx+1+t)+(yidx+t)]-=p3;
            acdvalue=_mm512_mask_sub_ps(vsetf(0.0),mif_t,acdvalue,p3);
            _mm512_mask_i32scatter_ps(a_cuda,mif_t,aidx,acdvalue,sizeof(float));

            //if(yidx == 0){
            vmask mif2_t = _mm512_mask_cmp_epi32_mask(mif_t,yidx,vseti(0),_MM_CMPINT_EQ);
                //b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
                vfloat bcdvalue=_mm512_mask_i32gather_ps(bcdvalue, mif2_t, xidx1t, b_cuda, sizeof(float));
                //b1 = m_cuda[Size*(xidx+1+t)+(yidx+t)]
                vint bcidx=_mm512_mask_mullo_epi32(vseti(0),mif2_t,vseti(Size),xidx1t);
                bcidx =_mm512_mask_add_epi32(bcidx,mif2_t,bcidx,yidx);
                bcidx =_mm512_mask_add_epi32(bcidx,mif2_t,bcidx,vseti(t));
                vfloat b1=_mm512_mask_i32gather_ps(vsetf(0.0),mif2_t,bcidx,m_cuda,sizeof(float));
                //b1*b_cuda[t]
                b1=_mm512_mask_mul_ps(b1,mif2_t,b1,vsetf(b_cuda[t]));
                b1=_mm512_mask_sub_ps(b1,mif2_t,bcdvalue,b1);
                //final commit
               _mm512_mask_i32scatter_ps(b_cuda,mif2_t,xidx1t,b1,sizeof(float));

            //}
    //}
/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void Fan2(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old, 
/*MOD_MAN: Para Begin*/ float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_Fan2 *p=(P_PR_Fan2 *)malloc(num_of_pts_launch*sizeof(P_PR_Fan2));//array of object passed to each pthread
    
/*MOD_AUTO*/ //Fan2 configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_Fan2 ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    //ker_parameters.arr=arr; 
    ker_parameters.m_cuda=m_cuda;
    ker_parameters.a_cuda=a_cuda;
    ker_parameters.b_cuda=b_cuda;
    ker_parameters.Size=Size;
    ker_parameters.j1=j1;
    ker_parameters.t=t;

    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for Fan2       
    pthread_create(&threads[i], &pt_attr,Fan2_KERNEL, (void *)(p+i));        //create with affinity

    #include "zpart9.h"

    /*MOD_MAN*///free shared memory
    //for (unsigned i=0; i<num_of_pts_launch; i++) {
        //if(p[i].warpid==0){
            //free(p[i].temp_on_cuda); free(p[i].power_on_cuda); free(p[i].temp_t); 
        //}
    //}
    /*MOD_MAN*///Ends

    //Free data
    free(p);free(threads);free(barrier);


}
#endif
