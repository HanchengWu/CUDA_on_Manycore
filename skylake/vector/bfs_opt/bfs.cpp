#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

//PHI Headers
#define DEBUG 0
#define VERIFY 1
#include "phi_template.h"
//Ends

#define MAX_THREADS_PER_BLOCK 16 

int no_of_nodes;
int edge_list_size;
FILE *fp;

//INCLUDE HEADERS TO OFFLOADED TO PHI HERE
//Structure to hold a node information
struct Node {
	int starting;
	int no_of_edges;
};

//include all kernel.h here
#include "k1k2R.h"

int affinity;

void Usage(int argc, char**argv){
    fprintf(stderr,"Usage: %s <affinity> <input_file>\n", argv[0]);
}

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) {

  char *input_f;
	if(argc!=3){
    Usage(argc, argv);
    exit(0);
	}

	affinity = atoi(argv[1]);
	
	input_f = argv[2];
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp) {
		printf("Error Reading graph file\n");
		return;
	}

	FILE *fpout = fopen("bfs_skylake_vector_times.txt", "a+");
	if(!fpout) {
	  printf("Error Saving stats\n");
	  exit(1);
	}

	// if file is empty then give it headers
	fseek (fpout, 0, SEEK_END); //move pointer to the end of file
	if ( ftell(fpout) == 0 ) { // if the pos is 0, it is empty
	  fprintf(fpout,"%-10s, %-10s, %-10s, ", "Affinity", "Usage", "# nodes");

	  // application specific
	  //fprintf(fpout,"no_of_nodes, ");

	  fprintf(fpout,"%-20s\n", "time");
	}
	fseek(fpout, 0, SEEK_SET); // reset pointer to start of file

	int source = 0;

	//phiSetDevice(1);

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK) {
		//num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	int *h_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	int *h_graph_visited = (int*) malloc(sizeof(int)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=0;
		h_updating_graph_mask[i]=0;
		h_graph_visited[i]=0;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	//source=0; //testing code line

	//set the source node as true in the mask
	h_graph_mask[source]=1;
	h_graph_visited[source]=1;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    

	printf("Read File\n");

	//Copy the Node list to device memory
/*	Node* d_graph_nodes;
  phiMalloc( &d_graph_nodes, no_of_nodes );
  phiMemcpy( d_graph_nodes, h_graph_nodes, no_of_nodes, CpuToPhi );

	//Copy the Edge List to device Memory
	int* d_graph_edges;
  phiMalloc( &d_graph_edges, edge_list_size );
  phiMemcpy( d_graph_edges, h_graph_edges, edge_list_size, CpuToPhi );

	//Copy the Mask to device memory
	int* d_graph_mask;
  phiMalloc( &d_graph_mask, no_of_nodes );
  phiMemcpy( d_graph_mask, h_graph_mask, no_of_nodes, CpuToPhi );

	int* d_updating_graph_mask;
	phiMalloc( &d_updating_graph_mask, no_of_nodes) ;
	phiMemcpy( d_updating_graph_mask, h_updating_graph_mask, no_of_nodes, CpuToPhi) ;

	//Copy the Visited nodes array to device memory
	int* d_graph_visited;
	phiMalloc( &d_graph_visited, no_of_nodes) ;
	phiMemcpy( d_graph_visited, h_graph_visited, no_of_nodes, CpuToPhi) ;

*/

	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
/*	
	// allocate device memory for result
	int* d_cost;
	phiMalloc( &d_cost, no_of_nodes);
	phiMemcpy( d_cost, h_cost, no_of_nodes, CpuToPhi);
*/
	//make a bool to check if the execution is over
//	int *d_over;
//	phiMalloc( &d_over, 1);

	printf("Copied Everything to Xeon Phi memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	printf("Start traversing the tree\n");
	    double start_time = gettime_ms();

	k1k2R( affinity, 1, grid, threads, h_graph_nodes, h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost, no_of_nodes);

	//Call the Kernel untill all the elements of Frontier are not false
	//int tt=0;
	/*
	int stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop=0;
		phiMemcpy( d_over, &stop, 1, CpuToPhi) ;

		Kernel( 1, grid, threads, d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
		// check if kernel execution generated and error

		Kernel2( 1, grid, threads, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error

		phiMemcpy( &stop, d_over, 1, PhiToCpu) ;
		k++;
	}
	while(stop);
	*/

	//while(tt++<11);
	double end_time = gettime_ms();

    //printf("Compute time: %lf\n", (end_time - start_time));
  fprintf(fpout,"%-10d, %-10d, ", affinity, no_of_nodes);
  //fprintf(fpout,"%d ", no_of_nodes);
  fprintf(fpout,"%-20lf\n", (end_time-start_time));


	printf("Kernel Executed %d times\n",k);

	// copy result from device to host
	//phiMemcpy( h_cost, d_cost, no_of_nodes, PhiToCpu);

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");

	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);

	//phiFree(d_graph_nodes);
	//phiFree(d_graph_edges);
	//phiFree(d_graph_mask);
	//phiFree(d_updating_graph_mask);
	//phiFree(d_graph_visited);
	//phiFree(d_cost);
}

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {
    no_of_nodes=0;
    edge_list_size=0;
    BFSGraph( argc, argv);
}
