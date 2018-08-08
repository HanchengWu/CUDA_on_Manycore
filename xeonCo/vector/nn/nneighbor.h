//Replace "nneighbor" with your "KERNEL NAME"
//Modify places marked with "MOD_MAN"

#ifndef __nneighbor__
#define __nneighbor__

#pragma offload_attribute (push, target(mic))
//Struct holds kernel config and info used for synchronization
typedef struct{
    #include "zpart1.h" //device configuration variables
    
    /*MOD_MAN*/  //define kernel parameters here for Kernel nneighbor
    int *guess_a;
    float *bestdis_a;
    int *query_a;
    int *vtxarr;
    int *edgearr;
    int *in_o;
    unsigned *deptharr;
    int dim;
    int objnum;
    int QS;
    int root;
} PR_nneighbor;

//Defines the object passed to each pthread instance for nneighbor
typedef struct {
    
    #include "zpart2.h" //pthread/block group variables 
    /*MOD_MAN*///shared mem pointer
    //int *s_mem;

/*MOD_AUTO*/ //kernel specific parameter object
    PR_nneighbor *kp;//define pointer to PR_nneighbor

} P_PR_nneighbor;

/*MOD_AUTO*/
void *nneighbor_KERNEL(void *arg);
#pragma offload_attribute (pop)

__attribute__ (( target(mic) )) __inline__ int vint2vfloat(__m512i *vi, __m512 *vf){
        float *ftmp=(float *)vf; int *itmp=(int *)vi;
        for (int i=0;i<16;i++) {
            (*ftmp++)=(float)(*itmp++);
        } 
}

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

__attribute__ (( target(mic) )) int nn_rec(__m512i curr, int depth, __m512i query, int *guess, float *bestDist, __m512i idx, \
                                            int dim, int objnum, int *vtxarr, int *edgearr, unsigned *deptharr, int *in_o, __mmask16 oldmk){

    //printf("\nargument passed!\n");
    //if ( UNDEFINED==curr )
    //      return 0;
    __mmask16 mk = _mm512_mask_cmp_epi32_mask(oldmk, curr, _mm512_set1_epi32(UNDEFINED), _MM_CMPINT_NE); //continue only if not equal
    int mk_int = _mm512_mask2int(mk); //check if any bit is 1
    if (mk_int == 0) {  /*printf("\ncheck mk: %d\n", mk_int);*/ 
        return 0; 
    }//return when all bits in mask is 0

    __m512i e_p = _mm512_mask_i32gather_epi32(e_p, mk, curr, vtxarr, 4); // int e_p=vtxarr[curr];
    __m512i left = _mm512_mask_i32gather_epi32(left, mk, e_p, edgearr, 4); // int left=edgearr[e_p];
    __m512i e_p_1 = _mm512_mask_add_epi32(e_p_1, mk ,e_p, _mm512_set1_epi32(1));// e_p_1= e_p+1;
    __m512i right = _mm512_mask_i32gather_epi32(right, mk, e_p_1, edgearr, 4); // int right=edgearr[e_p+1];

    __m512 dis_sq = _mm512_setzero_ps(); // float dis_sq=.0;
    //printf("\nvalue of mk:%d\n", _mm512_mask2int(mk));
    //printiv(query,"query");

    for (int i=0; i<dim; ++i){

        int *addr=in_o+i*objnum; //this was in readvalue before
        __m512i vcurr, vquery, diff,tmp;  
        vcurr = _mm512_mask_i32gather_epi32(vcurr, mk, curr, addr, 4); //readvalue(i,curr)
        vquery = _mm512_mask_i32gather_epi32(vquery, mk, query, addr, 4); //readvalue(i,curr)
        diff = _mm512_mask_sub_epi32(diff, mk, vcurr, vquery); //readvalue(i,curr)-readvalue(i,query);
    //  printiv(diff,"diff");
        diff = _mm512_mask_mullo_epi32(diff, mk, diff, diff); //diff * diff
    //  printiv(diff,"diff*diff");
        __m512 f_diff;
    //  f_diff = _mm512_mask_cvtfxpnt_round_adjustepi32_ps(f_diff, mk, diff, _MM_FROUND_TO_ZERO ,_MM_EXPADJ_NONE); // change diff*diff to float
        vint2vfloat(&diff,&f_diff);
//      f_diff=_mm512_castsi512_ps(diff);        
    //  printfv(f_diff,"f_diff");
        dis_sq = _mm512_mask_add_ps(dis_sq, mk, dis_sq, f_diff); // dis_sq+=tmp*tmp;
    //  printfv(dis_sq,"dis_sq");
    }

    __m512 dis = _mm512_mask_sqrt_ps( _mm512_setzero_ps(), mk, dis_sq); //  float dis=sqrt(dis_sq);
    //printfv(dis, "dis");

    //create new mask for if (dis<bestDist[idx] && dis !=0)
    __mmask16 mkl = _mm512_mask_cmp_ps_mask(mk, dis, _mm512_mask_i32gather_ps(dis, mk, idx, bestDist, 4) , _CMP_LT_OS); //dis<bestDist[idx]
    //printf("\nvalue of mkl:%d\n", _mm512_mask2int(mkl));

    __mmask16 mkr = _mm512_mask_cmp_ps_mask(mk, dis, _mm512_setzero_ps(), _CMP_NEQ_UQ);// dis!=0
    //printf("\nvalue of mkr:%d\n", _mm512_mask2int(mkr));

    __mmask16 mkif = _mm512_kand(mkl, mkr); //  the mask for if (dis<bestDist[idx] && dis !=0)
    //printf("\nvalue of mkif:%d\n", _mm512_mask2int(mkif));
    //below use mkif
    _mm512_mask_i32scatter_ps(bestDist, mkif, idx, dis, 4); //bestDist[idx] = dis;
    _mm512_mask_i32scatter_epi32(guess, mkif, idx, curr, 4); //guess[idx] = curr;
 
    //below resumes using maks mk
    //__m512i dim_l;    dim_l = _mm512_mask_rem_epi32(dim_l, mk, depth, _mm512_set1_epi32(dim));
    //printiv(dim_l,"dim_l");
     int dim_l=depth%dim;
     int *addr=in_o+dim_l*objnum; //this was in readvalue before
     __m512i cdim,qdim;
     cdim = _mm512_mask_i32gather_epi32(cdim, mk, curr, addr, 4); //readvalue(i,curr)
     qdim = _mm512_mask_i32gather_epi32(qdim, mk, query, addr, 4); //readvalue(i,curr)
    //printf("\nvalue of mk:%d\n", _mm512_mask2int(mk));

     //printiv(qdim,"qdim");
     //printiv(cdim,"cdim");
     __mmask16 mktru = _mm512_mask_cmp_epi32_mask(mk, qdim, cdim, _MM_CMPINT_LT);
     __mmask16 mkfal = _mm512_mask_cmp_epi32_mask(mk, qdim, cdim, _MM_CMPINT_NLT);
    //printf("\nvalue of mktru:%d\n", _mm512_mask2int(mktru));
    //printf("\nvalue of mkfal:%d\n", _mm512_mask2int(mkfal));

     //true branch
     nn_rec(left, depth+1, query, guess, bestDist, idx, dim, objnum, vtxarr, edgearr, deptharr, in_o, mktru);
        __m512i cdim_qdim = _mm512_mask_sub_epi32(cdim_qdim, mktru, cdim, qdim);
        __m512  fcq; vint2vfloat(&cdim_qdim, &fcq);
        __m512  bestDist_true = _mm512_mask_i32gather_ps(bestDist_true, mktru, idx, bestDist, 4);
        __mmask16 mktru_if = _mm512_mask_cmp_ps_mask(mktru, fcq, bestDist_true, _CMP_LT_OS);
            nn_rec(right, depth+1, query, guess, bestDist, idx, dim, objnum, vtxarr, edgearr, deptharr, in_o, mktru_if);

     //false branch
     nn_rec(right, depth+1, query, guess, bestDist, idx, dim, objnum, vtxarr, edgearr, deptharr, in_o, mkfal);
        __m512i qdim_cdim = _mm512_mask_sub_epi32(qdim_cdim, mkfal, qdim, cdim);
        __m512 fqc; vint2vfloat(&qdim_cdim, &fqc);
        __m512  bestDist_false = _mm512_mask_i32gather_ps(bestDist_false, mkfal, idx, bestDist, 4);
        __mmask16 mkfal_if = _mm512_mask_cmp_ps_mask(mkfal, fqc, bestDist_false, _CMP_LT_OS);
            nn_rec(left, depth+1, query, guess, bestDist, idx, dim, objnum, vtxarr, edgearr, deptharr, in_o, mkfal_if);

     return 0;
}

void *nneighbor_KERNEL(void *arg)
{
    /*MOD_AUTO*/
    P_PR_nneighbor *p=(P_PR_nneighbor *)arg;    //restore pthead data pointer
    PR_nneighbor *kp=p->kp;   //restore nneighbor config and para pointer
    
    /*MOD_MAN*///restore kernel nneighbor's parameters
    //int para=kp->para;
    int *guess_a=kp->guess_a;
    float *bestdis_a=kp->bestdis_a;
    int *query_a=kp->query_a;
    int *vtxarr=kp->vtxarr;
    int *edgearr=kp->edgearr;
    int *in_o=kp->in_o;
    unsigned *deptharr=kp->deptharr;
    int dim=kp->dim;
    int objnum=kp->objnum;
    int QS=kp->QS;
    int root=kp->root;

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
        vint _i= _mm512_mask_add_epi32(vseti(0),init_mask,vseti(blockDim.x*blockIdx.x),threadIdxx);
        //num of total threads
        __m512i numTotalThreads = _mm512_set1_epi32(gridDim.x*blockDim.x);
        //size of query queue
        __m512i _QS = _mm512_set1_epi32(QS);

        //if global index < queue size
        __mmask16 mk = _mm512_mask_cmp_epi32_mask(init_mask, _i, _QS, _MM_CMPINT_LT);
        int mk_int = _mm512_mask2int(mk);
        //int test=0;
        while (mk_int!=0){
            /////
            _mm512_mask_i32scatter_epi32(guess_a, mk, _i, _mm512_set1_epi32(UNDEFINED),4); //guess_a[i]=UNDEFINED;
            _mm512_mask_i32scatter_ps(bestdis_a,mk,_i,_mm512_set1_ps(std::numeric_limits<float>::max()),4); //bestdis_a[i]=std::numeric_limits<float>::max();
            __m512i _query;
            _query = _mm512_mask_i32gather_epi32(_query, mk, _i, query_a, 4);//query=query_a[i];            
            //printiv(_query,"_query");
            nn_rec(_mm512_set1_epi32(root), 0, _query, guess_a, bestdis_a, _i, dim, objnum, vtxarr, edgearr, deptharr, in_o, mk);//nn_rec(root,0,query,guess_a,bestdis_a,i);
            //printf("%d ",test++);
            //change to +numallthreads
            _i = _mm512_mask_add_epi32( vseti(0), mk, _i,numTotalThreads); //i += num of total cuda threads
            //update loop condition 
            mk = _mm512_mask_cmp_epi32_mask(mk, _i, _QS, _MM_CMPINT_LT);//mask for next iteration
            mk_int = _mm512_mask2int(mk);//convert mask to int, continue if any bit is 1
        }

/*MOD_MAN*///KERNEL ENDS//
        #include "zpart4.h" //contains debug code
    }
    return (NULL);
}

/*MOD_AUTO*/
void nneighbor(unsigned numPtsPerCore, float usage,const dim3 &dimGrid_old, const dim3 &dimBlock_old,
                 /*MOD_MAN: Para Begin*/
                int *guess_a,
                float *bestdis_a,
                int *query_a,
                int *vtxarr,
                int *edgearr,
                int *in_o,
                unsigned *deptharr,
                int dim,
                int objnum,
                int QS,
                int root
                 /*MOD_MAN: Para End*/)
{
#include "zpart5.h"

//Scalar Variables are copied in with "in( scalar: ONCE)"  
//Pointer Variables are copied in with "in(pointer: length(0) REUSE_A)". All poninters should point to Xeon Phi Device Memory    
#pragma offload target(mic: DEV_NUM) in(numPtsPerCore: ONCE) in(usage:ONCE) in(dimGrid: ONCE) in(dimBlock: ONCE) \
                in( guess_a: length(0) REUSE_A) \
                in( bestdis_a: length(0) REUSE_A) \
                in( query_a: length(0) REUSE_A) \
                in( vtxarr: length(0) REUSE_A) \
                in( edgearr: length(0) REUSE_A ) \
                in( in_o: length(0) REUSE_A) \
                in( deptharr: length(0) REUSE_A) \
                in( dim: ONCE ) \
                in( objnum: ONCE) \
                in( QS: ONCE ) \
                in( root: ONCE )
/*MOD_MAN Para*/     //           in( arr: length(0) REUSE_A   ) \
/*MOD_MAN Para*/     //           in( num: ONCE  ) 
    {//offload begins
    //Below calculates pthreads configuration
    #include "zpart6.h"

/*MOD_AUTO*/
    //array of objects passed to each pthread
    P_PR_nneighbor *p=(P_PR_nneighbor *)malloc(num_of_pts_launch*sizeof(P_PR_nneighbor));//array of object passed to each pthread
    
/*MOD_AUTO*/ //nneighbor configaration data and parameters //Kernel configuration<<<B,T>>> //# of concurrent blocks
    PR_nneighbor ker_parameters;ker_parameters.gridDim=dimGrid; ker_parameters.blockDim=dimBlock; ker_parameters.numActiveBlocks=numActiveBlocks;

/*MOD_MAN*/ //Kernel Parameters/Arguments are assigned here
    ker_parameters.guess_a=guess_a; 
    ker_parameters.bestdis_a=bestdis_a;
    ker_parameters.query_a=query_a;
    ker_parameters.vtxarr=vtxarr;
    ker_parameters.edgearr=edgearr;
    ker_parameters.in_o=in_o;
    ker_parameters.deptharr=deptharr;
    ker_parameters.dim=dim;
    ker_parameters.objnum=objnum;
    ker_parameters.QS=QS;
    ker_parameters.root=root;
    #include "zpart7.h"

/*MOD_MAN*/
    //USE FOLLOWING SECTION IF SHREAD MEM IS DEFINED

    //SHARED MEM ENDS
        
    #include "zpart8.h"

/*MOD_AUTO*/ //create pthreads for nneighbor       
    pthread_create(&threads[i], &pt_attr,nneighbor_KERNEL, (void *)(p+i));        //create with affinity

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
