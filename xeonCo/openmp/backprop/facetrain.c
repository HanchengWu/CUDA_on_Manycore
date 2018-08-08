#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();
//extern unsigned NUM_THEAD;

unsigned NUM_THREAD = 0; //OpenMP threads

int layer_size = 0;

backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_kernel(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("Training done\n");
}

int setup(argc, argv)
int argc;
char *argv[];
{
  if(argc!=3) {
    fprintf(stderr, "usage: backprop <num of threads> <num of input elements>\n");
    //fprintf(stderr, "usage: backprop <num of input elements>\n");
    exit(0);
  }


  NUM_THREAD = atoi(argv[1]);
  layer_size = atoi(argv[2]);
  
  int seed;

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
