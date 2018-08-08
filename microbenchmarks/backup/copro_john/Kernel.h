//Replace "Kernel" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __Kernel__
#define __Kernel__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel Kernel
    //int para;
    int *d_array;
    int sum;

} PR_Kernel;

//Defines the object passed to each pthread instance for Kernel
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_Kernel *kp;//define pointer to PR_Kernel

} P_PR_Kernel;

/*MOD_AUTO*/
void *Kernel_KERNEL(void *arg);

__attribute__ (( target(mic) )) int printfv( __m512 v, char *s) {
	float *tem =(float *)&v;
	printf("\n%s:\n",s);
	for (int i=0; i<16;i++,tem++){
				printf("%f, ",*tem);
	}
		printf("\n");
}

__attribute__ (( target(mic) )) int printiv( __m512i v,char *s) {
	int *tem =(int *)&v;
	printf("\n%s:\n",s);
	for (int i=0; i<16;i++,tem++){
				printf("%d, ",*tem);
	}
	printf("\n");
}

#pragma offload_attribute (pop)

void *Kernel_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_Kernel *p=(P_PR_Kernel *)arg;    //restore pthead data pointer
    PR_Kernel *kp=p->kp;   //restore Kernel config and para pointer
    
    /*MOD_MAN*///restore kernel Kernel's parameters
    //int para=kp->para;
    int *d_array = kp->d_array;
    int sum = kp->sum;

    #include "zpart3.h"    //recover kernel configuration

    /*MOD_MAN*/ //Activate corresponding threadIdx, all vector-lane variant variables
    vint threadIdxx= _mm512_mask_rem_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x)) ;
    vint threadIdxy= _mm512_mask_rem_epi32(vseti(0),init_mask,_mm512_div_epi32(_threadIdx, vseti(blockDim.x)), vseti(blockDim.y)) ;
    vint threadIdxz= _mm512_mask_div_epi32(vseti(0),init_mask,_threadIdx, vseti(blockDim.x*blockDim.y)) ;

    for(unsigned _bidx=p->blockgroupid; _bidx<numBlocksPerGrid; _bidx += kp->numActiveBlocks){
        /*MOD_MAN*/ //Activate corresponding blockIdx, all vector-lane invariant variables
        Dim3 blockIdx;
        blockIdx.x = _bidx%gridDim.x; 
        //blockIdx.y = (_bidx/gridDim.x)%gridDim.y;
        //blockIdx.z = _bidx/(gridDim.x*gridDim.y);
/*MOD_MAN*///KERNEL STARTS//

        vint tid = _mm512_mask_add_epi32( _mm512_set1_epi32(0), init_mask, _mm512_set1_epi32(blockIdx.x*16), threadIdxx ); 

        vint mem_values = vseti(0);

        for ( int loop_i = 0; loop_i < 10; loop_i++ ) {
          vint local_sum = vseti(0);
          printiv(tid, "tid");

          for( int mem_i = 0; mem_i < 500000000; mem_i+=32 ) {
            mem_values = _mm512_i32gather_epi32( _mm512_set1_epi32(mem_i), d_array, 1 ); 
            local_sum = _mm512_add_epi32( local_sum, mem_values );
          }

          printiv(local_sum, "local_sum");
        }

/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void Kernel(unsigned numPtsPerCore, float usage, const dim3 &dimGrid_old, const dim3 &dimBlock_old /*MOD_MAN: Para Begin*/ , int *d_array, int sum /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(usage:ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
            in( d_array: length(0) REUSE_A ) \
            in( sum: ONCE )
/*MOD_MAN Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN Para*/     //           in( num: ONCE  ) 
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_Kernel *p=(P_PR_Kernel *)malloc(num_of_pts_launch*sizeof(P_PR_Kernel));//array of object passed to each pthread
    
/*MOD_AUTO*/ //Kernel configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_Kernel ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    //ker_parameters.arr=arr; 
    ker_parameters.d_array = d_array;
    ker_parameters.sum = sum;
    
    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for Kernel       
    pthread_create(&threads[i], &pt_attr,Kernel_KERNEL, (void *)(p+i));        //create with affinity

    #include "zpart9.h"

    free(p); free(threads); free(barrier);
    }//offload ends

}
#endif
