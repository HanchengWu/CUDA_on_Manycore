#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern int setup(int argc, char** argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

extern unsigned NUM_THREAD;


double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

inline double gettime_ms() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return (t.tv_sec+t.tv_usec*1e-6)*1000;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern unsigned NUM_THEAD;


void bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   


  FILE *fpout = fopen("backprop_omp_times.txt","a+");
  if(!fpout) {
    printf("Error Saving stats\n");
    exit(1);
  }

  // if file is empty then give it headers
  fseek (fpout, 0, SEEK_END); //move pointer to the end of file
  if ( ftell(fpout) == 0 ) { // if the pos is 0, it is empty
    fprintf(fpout,"%-10s, ", "#Threads");

    // application specific
    //fprintf(fpout,"rows, cols, height, ");

    fprintf(fpout,"%-20s\n", "time");
  }
  fseek(fpout, 0, SEEK_SET); // reset pointer to start of file

  printf("Performing CPU computation\n");

  double total_time=0.0;
  double start_time = gettime_ms();

  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

  double end_time = gettime_ms();
  total_time+=end_time-start_time;


  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  start_time = gettime_ms();

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

  end_time = gettime_ms();
  total_time+=end_time-start_time;

  //printf("Compute time: %lf\n", total_time);
  fprintf(fpout,"%-10d, ", NUM_THREAD);
  //fprintf(fpout,"%d ", layer_size);
  fprintf(fpout,"%-20lf\n", total_time);



}
