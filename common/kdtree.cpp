#include "kdtree.h"
#include <limits>
#include <math.h>
#include <string.h>
#define ASIZE (64)

using namespace std;

int * kdtree::return_in_o(void){
        return in_o;
}

int * kdtree::return_vtxarr(void){
        return vtxarr;
}

int * kdtree::return_edgearr(void){
        return edgearr;
}

unsigned * kdtree::return_deptharr(void){
        return deptharr;
}


kdtree::kdtree(void){
	treeset=false;
	inputread=false;
	treebuild=false;
	build_balanced_tree=false;
}

kdtree::~kdtree(void){
	if (treeset)	_mm_free(in_o);
	if (treebuild){
		_mm_free(vtxarr);
		_mm_free(edgearr);
		_mm_free(deptharr);
	}
}

int kdtree::setvalue(int value, int dim, int obj){
	in_o[dim*OBN+obj]=value;
	return 0;
}

int kdtree::setbalance(bool t){
	build_balanced_tree=t;
	return t;
}

int kdtree::readvalue(int dim, int obj){
	return in_o[dim*OBN+obj];
}

int kdtree::set_tree(int dimension, int number){
	DIM=dimension;
	OBN=number;
	in_o=(int *)_mm_malloc(sizeof(int)*DIM*OBN,ASIZE);
	treeset=true;
	return 0;
}

int kdtree::read_cov(const char* in_file){
	if (!treeset){
		printf("\nTree parameters not set!\n");
		return -1;
	}

	char *buff=(char *)malloc(100*1024*1024);
	FILE *fp;
	if ( fp = fopen (in_file,"r") ){
                printf("\nFile open successful.\n");
    }else {
                perror ("Error opening file.\n");
                return 1;
    }
    /*
    for (int i=0;i<OBN ;++i) {
    	int dummy;
		if(fscanf(fp, "%d", &dummy) != 1) {
			fprintf(stderr, "Input file not large enough.\n");
			exit(1);
		}
		for(int j = 0; j < DIM; j++) {
			float tmp;
			if(fscanf(fp, "%d", &tmp) != 1) {
				fprintf(stderr, "Input file not large enough.\n");
				exit(1);
			}
			setvalue(tmp,j,i);
		}
    }*/

        char *p;
        for (int i=0;i<OBN ;++i) {
                fgets(buff, BUFF_SIZE, fp);
                p=strtok( buff , "," );
		setvalue(atoi(p)%500,0,i);
                for (int j=1;j<DIM;++j){
                        p=strtok( NULL , "," );
                       setvalue(atoi(p)%500,j,i);
                }
        }

        printf("successfully parsed.\n");
	fclose(fp);
//	free(buff);
	inputread=true;	
	return 0;
}

int kdtree::get_root(void){
	return root;
}

int kdtree::print_vertex_arr(void){
	printf("\nVertex Array:\n");
	for (int i=0;i<20;i++)
        	printf("%d ",vtxarr[i]);        
	printf("\n");
	return 0;
}

int kdtree::print_edge_arr(void){
	printf("\nEdge Array:\n");
	for (int i=0;i<40;i++)
	        printf("%d ",edgearr[i]);        
	printf("\n");
	return 0;
}

int kdtree::build_tree(void){
	if (!inputread) {
		printf("\nInput data not read!\n");
		return -1;
	}
	//DEBUG
	max_depth=0;
	max_node=0;
	vtxarr=(int *)_mm_malloc(OBN*sizeof(int),ASIZE);
	deptharr=(unsigned *)_mm_malloc(OBN*sizeof(unsigned),ASIZE);
    edgearr=(int *)_mm_malloc(2*OBN*sizeof(int),ASIZE);
	vec=(int *)_mm_malloc(sizeof(int)*OBN,ASIZE);
	edge_p=0;
	for (int i=0;i<OBN;++i){
                vec[i]=i;
        }
	if(build_balanced_tree){
		printf("Building balanced tree!\n");
		root=build_tree_balanced(0,OBN,0);
	}else{
		printf("Building unbalanced tree!\n");
		build_tree_unbalanced();	
	}
	printf("Tree successfully build!\n");
	_mm_free(vec);
	treebuild=true;
	return 0;
}

int kdtree::nn_single(int curr, int depth, int query, int *p, int *lor){
	if( UNDEFINED==curr )
		return 0;
	*p=curr;
	int e_p=vtxarr[curr];
	int left=edgearr[e_p];
	int right=edgearr[e_p+1];
	
	int dim=depth%DIM;
	//DEBUG
	if(depth > max_depth) {
		max_depth=depth;
		max_node=query;
	}
	int cdim=readvalue(dim,curr);
	int qdim=readvalue(dim,query);
	//if ai<curri
	if (qdim < cdim) {//insert to the left
		*lor=1;
		nn_single(left,depth+1,query,p,lor);
	}else{ //insert to the right
		*lor=2;
		nn_single(right,depth+1,query,p,lor);
	}
	return 0;
}

int kdtree::build_tree_unbalanced(void){
	root=0;

	vtxarr[root]=edge_p;
	edgearr[edge_p]=UNDEFINED;
	edgearr[edge_p+1]=UNDEFINED;
	edge_p+=2;

	int parent;
	int LOR;//left of right
	for(int i=1;i<OBN;++i){

		vtxarr[i]=edge_p;
		edgearr[edge_p]=UNDEFINED;
		edgearr[edge_p+1]=UNDEFINED;
		edge_p+=2;
		//look for i's parent
		parent=UNDEFINED;
		nn_single(root,0,i,&parent,&LOR);
		if (UNDEFINED==parent) {printf("\nError building.A\n"); exit(-1);}
		int t_e=vtxarr[parent];
		if(1==LOR)
			edgearr[t_e]=i;
		else if(2==LOR)
			edgearr[t_e+1]=i;
		else {printf("\nError building.B\n"); exit(-1);}

	}
		
	return 0;

}
int *d_val;
int compare_point(const void *a, const void *b) {
	if(d_val[*((int *)a)] < d_val[*((int *)b)]) {
		return -1;
	} else if(d_val[*((int *)a)] > d_val[*((int *)b)]) {
		return 1;
	} else {
		return 0;
	}
}
int kdtree::build_tree_balanced(int start, int length, int depth){

        if (length == 0)
                return UNDEFINED;
        if (depth > max_depth) {
        	max_depth=depth;
        }
        if (length ==1 ){
                
		int tmp=*(vec+start);
                vtxarr[tmp]=edge_p;
		deptharr[tmp]=depth;
                edgearr[edge_p]=UNDEFINED;
                edgearr[edge_p+1]=UNDEFINED;
                edge_p+=2;
                return tmp;
        }
        int dim = depth%DIM;

        int *c_vec=vec+start;//position of vector[start]=&vector[start]
        d_val=in_o+dim*OBN;//set the d_val and dim for function compare_point
        qsort(c_vec,length,sizeof(int),compare_point);

        int middle = length / 2;

        while (  (middle!=0) &&  (d_val[c_vec[middle-1]]>=d_val[c_vec[middle]])  )  {
                middle--;
        }

        int pivot = c_vec[middle];
        int left=build_tree_balanced(start, middle, depth+1);
        int right=build_tree_balanced(start+middle+1,length-middle-1,depth+1);
        vtxarr[pivot]=edge_p;
	deptharr[pivot]=depth;
        edgearr[edge_p]=left;
        edgearr[edge_p+1]=right;
        edge_p+=2;
        return pivot;
}

int *vec_i;
void kdtree::sort_input_rec(int start, int length, int depth){

		if (length == 0)
                return;

        if (length ==1 ){

                return;
        }
        int dim = depth%DIM;

        int *c_vec=vec_i+start;//position of vector[start]=&vector[start]
        d_val=in_o+dim*OBN;//set the d_val and dim for function compare_point
        qsort(c_vec,length,sizeof(int),compare_point);

        int middle = length / 2;

        while (  (middle!=0) &&  (d_val[c_vec[middle-1]]>=d_val[c_vec[middle]])  )  {
                middle--;
        }

        sort_input_rec(start, middle, depth+1);
        sort_input_rec(start+middle+1,length-middle-1, depth+1);
        return;
}
int idx;
void kdtree::sort_input(int *ip, int number){
	vec_i = ip;
	if(build_balanced_tree) {sort_input_rec(0,number,0);}
	else {
		idx=0;
		traverse_tree(root);
		printf("\nsorted %d inputs\n",idx);
	}
}
void kdtree::traverse_tree(int nodeid){

	int edge_p=vtxarr[nodeid];
	int leftc=edgearr[edge_p];
	int rightc=edgearr[edge_p+1];
	if (leftc!=UNDEFINED) traverse_tree(leftc);
    vec_i[idx++]=nodeid;
	if(rightc!=UNDEFINED) traverse_tree(rightc);
}

int kdtree::nn_rec(int curr, int depth, int query, int *guess, float *bestDist){
	if( UNDEFINED==curr )
		return 0;
	int e_p=vtxarr[curr];
	int left=edgearr[e_p];
	int right=edgearr[e_p+1];
//update best distance
	float dis_sq=.0;

	for(int i=0;i<DIM;++i){
		float tmp=readvalue(i,curr)-readvalue(i,query);
		dis_sq+=tmp*tmp;
	}

	float dis=sqrt(dis_sq);

	if (dis<*bestDist && dis !=0){ 
//	if (dis<*bestDist ){ 
		*bestDist = dis;
		*guess = curr;
	}
	int dim=depth%DIM;
	int cdim=readvalue(dim,curr);
	int qdim=readvalue(dim,query);
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
//should we parallelize kD tree traversal in a different way like our tree height
int kdtree::nn(int *query, int *guess, float *bestDist, int len ){
	//see if root set, otherwise return
	//recursively call nn_rec

	for(int i=0;i<len;i++){	
		guess[i]=UNDEFINED;
		bestDist[i]=std::numeric_limits<float>::max();
		nn_rec(root,0,query[i],guess+i,bestDist+i);
//		printf("\nnn of query node %d is node %d.\n", i, guess[i]);
	}
//	printf("\nThe nearest distance is: %f.\n",bestDist);
//	printf("\nThe nearest node is node:%d.\n\n",g_guess);
	return 0;	
}

//just rand() didn't work now, result of rand() has to be rounded, probablly due to overflow
int kdtree::gnrt_random(void){
	if (!treeset){
		printf("\nTree parameters not set!\n");
		return -1;
	}

        for (int i=0;i<OBN ;++i) {
                for (int j=1;j<DIM;++j){
                       setvalue(rand()%500,j,i);
                }
        }
        printf("\nsuccessfully generate random data for nodes.\n");
	inputread=true;	
	return 0;
}


int kdtree::build_tree_from_txt(void){
	if(!treeset) {
		printf("\nThe tree parameters haven't been set yet. Cannot locate txt file.\n");
		return -1;
	}
	char filename[50]={};//long enough for current use

	char digit[20];
	sprintf(digit,"%d",OBN);
	//get filename ../Input/xxxxx.txt
	strcat(filename,"../Input/");
	strcat(filename,digit);
	strcat(filename,".txt");
        
	vtxarr=(int *)_mm_malloc(OBN*sizeof(int),ASIZE);
        deptharr=(unsigned *)_mm_malloc(OBN*sizeof(unsigned),ASIZE);
        edgearr=(int *)_mm_malloc(2*OBN*sizeof(int),ASIZE);

        FILE *fp;
        if (fp = fopen (filename,"r"))
                printf("\n%s open succ.\n",filename);
        else {
                printf("Error open file %s.\n",filename);
                exit(EXIT_FAILURE);
        }

        char *p;
        size_t bsize=100*1024*1024;//the size is not equal to the size of edge/depth/vtx array, cause an integer may need several chars to represent in txt, the size is set to 100M, which is big enough for structure file of tree with nodes less than 400000
        char *buff=(char *)malloc(bsize);

        fgets(buff,bsize,fp);
        p=strtok(buff," ");
        root=atoi(p);

	//printf("\nroot parsed\n");

        fgets(buff,bsize,fp);
        p=strtok(buff," ");
        vtxarr[0]=atoi(p);
        for(int i=1;i<OBN;i++){
	        p=strtok(NULL," ");
                vtxarr[i]=atoi(p);
        }
       //printf("\nvtxarr parsed\n");
        fgets(buff,bsize,fp);
        p=strtok(buff," ");
        deptharr[0]=atoi(p);
        for(int i=1;i<OBN;i++){
                p=strtok(NULL," ");
                deptharr[i]=atoi(p);
       }

	//printf("\ndepth arr parsed\n");	
	
	fgets(buff,bsize,fp);
        p=strtok(buff," ");
        edgearr[0]=atoi(p);

        for(int i=1;i<2*OBN;i++){
                p=strtok(NULL," ");
                edgearr[i]=atoi(p);
        }

	//printf("\nedge arr parsed.\n");
        treebuild=true;
        fclose(fp);
        free(buff);
        return 0;
}



int kdtree::save_built_tree_to_txt(void){//building a tree takes lot of time, this can be used to save built tree to ctree.txt file
	if(!treebuild) return -1;

	FILE *fp;

        if (fp = fopen ("ctree.txt","w"))
                printf("\nctree.txt open succ.\n");
        else {
                perror("Error open file.\n");
                return 1;
        }

        fprintf(fp,"%d ",root);

        fprintf(fp,"\n");

        for(int i=0;i<OBN;i++){
                fprintf(fp,"%d ",vtxarr[i]);
        }

        fprintf(fp,"\n");

        for(int i=0;i<OBN;i++){
                fprintf(fp,"%u ",deptharr[i]);
        }

        fprintf(fp,"\n");

        for(int i=0;i<2*OBN;i++){

                fprintf(fp,"%d ",edgearr[i]);
        }

        fprintf(fp,"\n");

        fclose(fp);

	printf("\nTree structure saved!\n");

	return 0;
}


	/*
	//bubble sorting elements in vec [start,......start+length)
        for(int i=0; i<length-1; i++){
                for(int j=0;j<length-1-i ;j++){
                        if ( d_val[c_vec[j]] > d_val[c_vec[j+1]] ) {
                                int tem=c_vec[j];
                                c_vec[j]=c_vec[j+1];
                                c_vec[j+1]=tem;
                        }
                }
        }
	//faster sorting?
*/

