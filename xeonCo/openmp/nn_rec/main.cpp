#include "kdtree.h"
#include <cfloat>

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
//#include <limits>

#define ALLOC   alloc_if(1)
#define FREE    free_if(1)
#define RETAIN  free_if(0)
#define REUSE   alloc_if(0)

#define DEV_NUM 0

inline double gettime_ms() {
        struct timeval t;
        gettimeofday(&t,NULL);
        return (t.tv_sec+t.tv_usec*1e-6)*1000;
}

#define numofthreads 30

bool verify=true;

#pragma offload_attribute (push, target(mic))
#include <limits>
int QS=64;	
int *vtxarr;
unsigned *deptharr;
int *edgearr;
int *in_o;
int *query_a;
int *guess_a;
float *bestdis_a;
int dim;
int objnum;
#pragma offload_attribute (pop)


__attribute__ (( target(mic) )) __inline__ int readvalue(int dim_l, int obj_l){
        return in_o[dim_l*objnum+obj_l];
}


__attribute__ (( target(mic) )) int nn_rec(int curr, int depth, int query, int *guess, float *bestDist){
	if( UNDEFINED==curr )
		return 0;
	int e_p=vtxarr[curr];
	int left=edgearr[e_p];
	int right=edgearr[e_p+1];
//update best distance
	float dis_sq=.0;

	for(int i=0;i<dim;++i){
		float tmp=readvalue(i,curr)-readvalue(i,query);
		dis_sq+=tmp*tmp;
	}

	float dis=sqrt(dis_sq);

	if (dis<*bestDist && dis !=0){ 
//	if (dis<*bestDist ){ 
		*bestDist = dis;
		*guess = curr;
	}
	int dim_l=depth%dim;
	int cdim=readvalue(dim_l,curr);
	int qdim=readvalue(dim_l,query);
	//if ai<curri
	if (qdim < cdim) {
		nn_rec(left,depth+1,query,guess,bestDist);
		if (cdim-qdim< *bestDist)
			nn_rec(right,depth+1,query,guess,bestDist);
	}else{ 
		nn_rec(right,depth+1,query,guess,bestDist);
		if (qdim-cdim< *bestDist)
			nn_rec(left,depth+1,query,guess,bestDist);
	}
	return 0;
}

int main(int argc, char *argv[]){
	if (5!= argc) {
		printf("Arguments is less then enough!\n");
		exit(0);
	}

	double time;
	
	dim=7;
	objnum=atoi(argv[1]);
	QS = objnum;
	int balanced=atoi(argv[2]);
	int random=atoi(argv[3]);
	int sortin=atoi(argv[4]);
	printf("Dim:%d. OBJNUM:%d. QueryNum:%d. Random:%d. Balanced tree:%d, Sorted:%d.\n",dim,objnum,QS,random,balanced, sortin);

	kdtree mytree;
	mytree.set_tree(dim,objnum);
	mytree.setbalance(balanced);
	if(random==0){
		mytree.read_cov("../Input/covtype.data");
	}else{
		mytree.gnrt_random();	
	}

// 	mytree.build_tree_from_txt();
 	mytree.build_tree();
	printf("The root node is:%d. Max depth is %d.\n", mytree.get_root(), mytree.max_depth );

	#pragma offload target(mic: DEV_NUM) 	
	{
		printf("\nInitialize PHI\n");
		fflush(0);
	}
	
	vtxarr=mytree.return_vtxarr();
	deptharr=mytree.return_deptharr();
	edgearr=mytree.return_edgearr();
	in_o=mytree.return_in_o();
	
	//allocate and copy queries related info to PHI
	query_a=(int *)malloc(QS*sizeof(int));
	guess_a=(int *)malloc(objnum*sizeof(int));
	bestdis_a=(float *)malloc(objnum*sizeof(float));
	
	for(int j=0;j<QS;++j){
                query_a[j]=j;
	}
	printf("\nQueries are initialized!\n");
	if(sortin==1){
		printf("queries are sorted!");
		mytree.sort_input(query_a,QS);
	}
	int root=mytree.get_root();
	//allocate data on mic
        #pragma offload_transfer target(mic: DEV_NUM) \
                nocopy( guess_a: length(objnum) ALLOC RETAIN ) \
                nocopy( bestdis_a: length(objnum) ALLOC RETAIN ) \
                nocopy( query_a: length(QS) ALLOC RETAIN ) \
                nocopy( vtxarr: length(objnum) ALLOC RETAIN ) \
                nocopy( edgearr: length(2*objnum) ALLOC RETAIN ) \
                nocopy( in_o: length(dim*objnum) ALLOC RETAIN ) \
                nocopy( deptharr: length(objnum) ALLOC RETAIN ) \
                nocopy( dim: ALLOC RETAIN ) \
                nocopy( objnum: ALLOC RETAIN) \
                nocopy( QS: ALLOC RETAIN ) \
                nocopy( root: ALLOC RETAIN ) 

	//transfer data to  mic
        #pragma offload_transfer target(mic: DEV_NUM) \
                nocopy( guess_a: length(objnum) REUSE RETAIN ) \
                nocopy( bestdis_a: length(objnum) REUSE RETAIN ) \
                in( query_a: length(QS) REUSE RETAIN ) \
                in( vtxarr: length(objnum) REUSE RETAIN ) \
                in( edgearr: length(2*objnum) REUSE RETAIN ) \
                in( in_o: length(dim*objnum) REUSE RETAIN ) \
                in( deptharr: length(objnum) REUSE RETAIN ) \
                in( dim: REUSE RETAIN ) \
                in( objnum: REUSE RETAIN) \
                in( QS: REUSE RETAIN ) \
                in( root: REUSE RETAIN )

	//transfer done
	printf("\nQueries begins!\n");
	time = gettime_ms();
	
	#pragma offload target(mic: DEV_NUM) \
                nocopy( guess_a: length(objnum) REUSE RETAIN ) \
                nocopy( bestdis_a: length(objnum) REUSE RETAIN ) \
                nocopy( query_a: length(QS) REUSE RETAIN ) \
             	nocopy( vtxarr: length(objnum) REUSE RETAIN ) \
                nocopy( edgearr: length(2*objnum) REUSE RETAIN ) \
                nocopy( in_o: length(dim*objnum) REUSE RETAIN ) \
                nocopy( deptharr: length(objnum) REUSE RETAIN ) \
                nocopy( dim: REUSE RETAIN ) \
                nocopy( objnum: REUSE RETAIN) \
                nocopy( QS: REUSE RETAIN ) \
                nocopy( root: REUSE RETAIN )	
	{//offload begins
	   #pragma omp parallel for shared(guess_a,bestdis_a,query_a,QS,root) num_threads(numofthreads) schedule(dynamic,10)
	   for(int i=0;i<QS;i++){	
		guess_a[i]=UNDEFINED;
		bestdis_a[i]=std::numeric_limits<float>::max();
		nn_rec(root,0,query_a[i],guess_a+i,bestdis_a+i);
	   }
	}//offload ends
	printf("\nQuerying on CPU-OpenMP-rec returns with time = %.2f ms.\n",gettime_ms()-time);

	//copy data back to CPU
	#pragma offload_transfer target(mic: DEV_NUM) \
            out( guess_a: length(objnum) REUSE FREE ) \
            out( bestdis_a: length(objnum) REUSE FREE ) 


	if(verify){
		int *guess_v=(int *)malloc(QS*sizeof(int));
		float *bestdis_v=(float *)malloc(QS*sizeof(float));
		printf("\nverification begins!\n");
		time = gettime_ms();
	    mytree.nn(query_a,guess_v,bestdis_v,QS);
	    printf("\nQuerying on CPU-verification returns with time = %.2f ms.\n",gettime_ms()-time);
	    printf("\n");
	
		for (int i=0;i<QS;++i){
		   float tem=fabs(bestdis_a[i]-bestdis_v[i]);
		   if(tem>0.0001){
	            printf("\nVerification failed at query %d. CPU-multi-rec bestdis %.5f. CPU bestdis %.5f. The diff is %.5f.\n",i,bestdis_a[i],bestdis_v[i],tem);
				free(guess_v);
				free(bestdis_v);
				return 1;
		   }
	    }
	
		free(guess_v);
		free(bestdis_v);
	    printf("\nVefication passed!\n");
	}
	
	free(query_a);
	free(bestdis_a);
	free(guess_a);	
	return 0;
}
