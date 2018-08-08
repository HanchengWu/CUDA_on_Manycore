//Replace "euclid" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __euclid__
#define __euclid__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel euclid
    LatLong *d_locations;
    float *d_distances; 
    int numRecords;
    float lat; 
    float lng;

} PR_euclid;

//Defines the object passed to each pthread instance for euclid
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_euclid *kp;//define pointer to PR_euclid

} P_PR_euclid;

/*MOD_AUTO*/
void *euclid_KERNEL(void *arg);

#pragma offload_attribute (pop)

void *euclid_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_euclid *p=(P_PR_euclid *)arg;    //restore pthead data pointer
    PR_euclid *kp=p->kp;   //restore euclid config and para pointer
    
    /*MOD_MAN*///restore kernel euclid's parameters
    LatLong *d_locations=kp->d_locations;
    float *d_distances=kp->d_distances;
    int numRecords=kp->numRecords;
    float lat=kp->lat;
    float lng=kp->lng;
   

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

        //global index
        vint gid= _mm512_mask_add_epi32(vseti(0),init_mask,vseti(blockDim.x*blockIdx.x),threadIdxx);
        //num of total threads
        __m512i numTs = _mm512_set1_epi32(gridDim.x*blockDim.x);
        //size of query queue
        __m512i v_numRecords = _mm512_set1_epi32(numRecords);

        //if gid < numRecords
        __mmask16 mk = _mm512_mask_cmp_epi32_mask(init_mask, gid, v_numRecords, _MM_CMPINT_LT);
        int mk_int = _mm512_mask2int(mk);

        vfloat fzero=vsetf(0.0);

        while (mk_int!=0){
            //vfloat mylat;
            float *add=(float *)d_locations;
            vfloat mylat = _mm512_mask_i32gather_ps(fzero, mk, gid, add, sizeof(LatLong));
            vfloat mylng = _mm512_mask_i32gather_ps(fzero, mk, gid, (add+1), sizeof(LatLong));

            //vfloat p1=lat-mylat;
            vfloat p1= _mm512_mask_sub_ps(fzero, mk, vsetf(lat), mylat);
            //vfloat p2=lng-mylng;
            vfloat p2= _mm512_mask_sub_ps(fzero, mk, vsetf(lng), mylng);
            //vfloat p3=p1*p1+p2*p2;
            vfloat p3= _mm512_mask_mul_ps(fzero, mk, p1, p1);
                   p3= _mm512_mask3_fmadd_ps(p2, p2, p3, mk); 
            //vfloat p4=sqrt(p3);
            vfloat p4= _mm512_mask_sqrt_ps(fzero,mk, p3);      
            //d_distances[globalId] = p4;
            _mm512_i32scatter_ps(d_distances, gid, p4, sizeof(float));

            //change to +numallthreads
            gid = _mm512_mask_add_epi32(vseti(0), mk, gid, numTs); //i += num of total cuda threads
            //update loop condition 
            mk = _mm512_mask_cmp_epi32_mask(mk, gid, v_numRecords, _MM_CMPINT_LT);//mask for next iteration
            mk_int = _mm512_mask2int(mk);//convert mask to int, continue if any bit is 1
        }


/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void euclid(unsigned numPtsPerCore, float usage,const dim3 &dimGrid_old, const dim3 &dimBlock_old,
             /*MOD_MAN: Para Begin*/
             LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng 
             /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
/*MOD_MAN Para*/                in(d_locations: length(0) REUSE_A) \
                                in(d_distances: length(0) REUSE_A) \
                                in(numRecords: ONCE) \
                                in(lat: ONCE) \
                                in(lng: ONCE)
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_euclid *p=(P_PR_euclid *)malloc(num_of_pts_launch*sizeof(P_PR_euclid));//array of object passed to each pthread
    
/*MOD_AUTO*/ //euclid configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_euclid ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    ker_parameters.d_locations=d_locations;
    ker_parameters.d_distances=d_distances;
    ker_parameters.numRecords=numRecords;
    ker_parameters.lat=lat;
    ker_parameters.lng=lng; 

    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED
    //if(p[i].warpid==0) {
    //  p[i].s_mem=(int *)malloc(sizeof(int)*numTdsPerBlock);
    //}
    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for euclid       
    pthread_create(&threads[i], &pt_attr,euclid_KERNEL, (void *)(p+i));        //create with affinity

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

    }//offload ends

}
#endif
