#include "kdtree.h"
#include <cfloat>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "phi_template.h"

#define DEBUG 0
#define VERIFY 0

#include <limits>

#include "nneighbor.h"

int QS=64;	
int dim;
int objnum;
int verify=0;
unsigned ptspercore=1;
float usage=1;


int main(int argc, char *argv[]){
	if (7!= argc) {
		printf("Arguments is less then enough!\n");
		exit(0);
	}

	double time;
	
	dim=7;
	objnum=atoi(argv[1]);
	QS = objnum;
	int random=atoi(argv[2]);
	int balanced=atoi(argv[3]);
	int sortin=atoi(argv[4]);
	ptspercore = atoi(argv[5]);
	usage = atof(argv[6]);
	printf("Dim:%d. OBJNUM:%d. QueryNum:%d. Random:%d. Balanced tree:%d, Sorted:%d.\n",dim,objnum,QS,random,balanced, sortin);

	kdtree mytree;
	mytree.set_tree(dim,objnum);
	mytree.setbalance(balanced);
	if(random==0){
		mytree.read_cov("../../Input/covtype.data");
	}else{
		mytree.gnrt_random();	
	}

// 	mytree.build_tree_from_txt();
 	mytree.build_tree();
	printf("The root node is:%d. Max depth is %d.\n", mytree.get_root(), mytree.max_depth );


	//retrive pointers from tree data	
	int *vtxarr=mytree.return_vtxarr();
	unsigned *deptharr=mytree.return_deptharr();
	int *edgearr=mytree.return_edgearr();
	int *in_o=mytree.return_in_o();

	int *guess=(int *)malloc(objnum*sizeof(int));
	float *bestdis=(float *)malloc(objnum*sizeof(float));
	int *query=(int *)malloc(QS*sizeof(int));

	for(int j=0;j<QS;++j){
                query[j]=j;
	}

	printf("\nQueries are initialized!\n");

	if(sortin==1){
		printf("queries are sorted!");
		mytree.sort_input(query,QS);
	}
	int root=mytree.get_root();

	int *d_guess;
	float *d_bestdis;
	int *d_query;
	int *d_vtxarr;
	int *d_edgearr;
	unsigned *d_deptharr;
	int *d_in_o;
	//allocate data on mic
	//transfer done
	printf("\nQueries begins!\n");

    FILE *fresult = fopen("result_pro.txt","a+");
	time = gettime_ms();
	nneighbor(ptspercore, usage ,240, 16, guess, bestdis, query, vtxarr, edgearr, in_o, deptharr, dim, objnum, QS, root);
	time = gettime_ms()-time;

    fprintf(fresult,"%-8d,%-10d,%-20lf\n",ptspercore,objnum, time);

	printf("\nQuerying on Phi-Vec-Rec returns with time = %.2f ms.\n",time);
    fclose(fresult);

	if(verify){
		int *guess_v=(int *)malloc(QS*sizeof(int));
		float *bestdis_v=(float *)malloc(QS*sizeof(float));
		printf("\nverification begins!\n");
		time = gettime_ms();
	    mytree.nn(query,guess_v,bestdis_v,QS);
	    printf("\nQuerying on CPU-verification returns with time = %.2f ms.\n",gettime_ms()-time);
	    printf("\n");
	
		for (int i=0;i<QS;++i){
		   float tem=fabs(bestdis[i]-bestdis_v[i]);
		   if(tem>0.0001){
	            printf("\nVerification failed at query %d. GPU-multi-rec bestdis %.5f. CPU bestdis %.5f. The diff is %.5f.\n",i,bestdis[i],bestdis_v[i],tem);
				free(guess_v);
				free(bestdis_v);
				return 1;
		   }
	    }
		free(guess_v);
		free(bestdis_v);
	    printf("\nVefication passed!\n");
	}

	free(guess);free(bestdis);free(query);
	return 0;
}
