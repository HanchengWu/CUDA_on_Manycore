#ifndef KDTREE_H_
#define KDTREE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFF_SIZE 512000
#define UNDEFINED 0xffffffff

class kdtree{
private:
	//parameters used to initialize the tree structure
	int DIM;//input data dimension
	int OBN;//nubmer of input objects
	//flags used internally
	bool treeset;
	bool inputread;
	bool treebuild;
	bool build_balanced_tree;
	//
	int *in_o;
	int root;
	int *vec;
	
	int *vtxarr;
	int *edgearr;
	unsigned *deptharr;	

	int edge_p;
	int build_tree_balanced(int start, int length, int depth);
	int build_tree_unbalanced(void);
	//followings are for nearest neighbor search
	int g_guess;
	float bestDist;
	int nn_single(int curr, int depth, int query, int *p, int *lor);
		
	int setvalue(int value, int dim, int obj);
	void sort_input_rec(int start, int length, int depth);
	void traverse_tree(int nodeid);

public:
	kdtree(void);
	~kdtree(void);
	int set_tree(int dimension,int number);
	int setbalance(bool);
	int read_cov(const char *in_file);
	int gnrt_random(void);
	int build_tree(void);
	int build_tree_from_txt(void);
	int save_built_tree_to_txt(void);

	int print_vertex_arr(void);
	int print_edge_arr(void);

	int nn_rec(int curr, int depth, int query, int *guess, float *bestDist);
	int nn(int *query, int *guess, float *bestDist,int qlen);
	int readvalue(int dimension, int objnum);

	int get_root(void);
	int *return_in_o(void);\
	int *return_vtxarr(void);
        int *return_edgearr(void);
        unsigned *return_deptharr(void);
    void sort_input(int *ip, int number);

	//DEBUG
 	int max_depth;
	int max_node;
}; 



#endif
