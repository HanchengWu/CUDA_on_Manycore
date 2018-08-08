
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// includes, kernels
#include "backprop.h"
#include <omp.h>

#define ABS(x) (((x) > 0.0) ? (x) : (-(x)))

int layer_size = 0;
int num_blocks = 0;
//unsigned int num_blocks = 0;

int affinity;
float usage;

#define DEBUG 0
#define VERIFY 0

#include "phi_template.h"
#include "bpnn_layerforward_CUDA.h"
#include "bpnn_adjust_weights_cuda.h"

// Return random number between 0.0 and 1.0 
float drnd() { return ((float)rand() / (float)BIGRND); }

// Return random number between -1.0 and 1.0
float dpn1() { return ((drnd() * 2.0) - 1.0); }

// The squashing function.  Currently, it's a sigmoid.

float squash(float x) {
    float m;
    // x = -x;
    // m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
    // return(1.0 / (1.0 + m));
    return (1.0 / (1.0 + exp(-x)));
}

void bpnn_randomize_weights(float **w, int m, int n) {
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
     w[i][j] = (float) rand()/RAND_MAX;
    }
  }
}

void bpnn_randomize_row(float *w, int m) {
	int i;
	for (i = 0; i <= m; i++) {
	 w[i] = 0.1;
  }
}

void bpnn_zero_weights(float **w, int m, int n) {
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}

void bpnn_initialize(int seed) {
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}

BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out) {
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units =  (float *)malloc((n_in + 1)*sizeof(float));
  newnet->hidden_units = (float *)malloc((n_hidden + 1)*sizeof(float));
  newnet->output_units = (float *)malloc((n_out + 1)*sizeof(float));

  newnet->hidden_delta = (float *)malloc((n_hidden + 1)*sizeof(float));
  newnet->output_delta = (float *)malloc((n_out + 1)*sizeof(float));
  newnet->target = (float *)malloc((n_out + 1)*sizeof(float));

  newnet->input_weights = (float**)malloc((n_in + 1)*sizeof(float*));
  for ( int i = 0; i < (n_in+1); i++ )
    newnet->input_weights[i] = (float *)malloc((n_hidden + 1)*sizeof(float));

  newnet->hidden_weights = (float**)malloc((n_hidden + 1)*sizeof(float*));
  for ( int i = 0; i < (n_hidden+1); i++ )
    newnet->hidden_weights[i] = (float *)malloc((n_out + 1)*sizeof(float));

  newnet->input_prev_weights = (float**)malloc((n_in + 1)*sizeof(float*));
  for ( int i = 0; i < (n_in+1); i++ )
    newnet->input_prev_weights[i] = (float *)malloc((n_hidden + 1)*sizeof(float));

  newnet->hidden_prev_weights = (float**)malloc((n_hidden + 1)*sizeof(float*));
  for ( int i = 0; i < (n_hidden+1); i++ )
    newnet->hidden_prev_weights[i] = (float *)malloc((n_out + 1)*sizeof(float));

  return (newnet);
}

void bpnn_free(BPNN *net) {
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *) net->input_units);
  free((char *) net->hidden_units);
  free((char *) net->output_units);

  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

  for (i = 0; i <= n1; i++) {
    free((char *) net->input_weights[i]);
    free((char *) net->input_prev_weights[i]);
  }
  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    free((char *) net->hidden_weights[i]);
    free((char *) net->hidden_prev_weights[i]);
  }
  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
}


BPNN *bpnn_create(int n_in, int n_hidden, int n_out) {

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return (newnet);
}

void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2) {
  float sum;
  int j, k;

  // Set up thresholding unit
  l1[0] = 1.0;
#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for shared(conn, n1, n2, l1) private(k, j) reduction(+: sum) schedule(static)
#endif 
  // For each unit in second layer
  for (j = 1; j <= n2; j++) {
    // Compute weighted sum of its inputs 
    sum = 0.0;
    for (k = 0; k <= n1; k++) {	
      sum += conn[k][j] * l1[k]; 
    }
    l2[j] = squash(sum);
  }
}

void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err) {
  int j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}

void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err) {
  int j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}

void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw) {
  float new_dw;
  int k, j;
  ly[0] = 1.0;

#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for  \
      shared(oldw, w, delta) \
	  private(j, k, new_dw) \
	  firstprivate(ndelta, nly) 
#endif 
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
	  w[k][j] += new_dw;
	  oldw[k][j] = new_dw;
    }
  }
}

void bpnn_print(BPNN *net) {
  int n1, n2, n3, i, j;
  float dvalue, **w;

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  printf("%dx%dx%d network\n", n1, n2, n3);

  w = net->input_weights;

  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w[i][j];
      printf("%lf, ", dvalue);
    }
  }

  printf("\n\n\n");

  w = net->hidden_weights;
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w[i][j];
      printf("%lf, ", dvalue);
    }
  }
}

void bpnn_train_cuda(BPNN *net, float *eo, float *eh) {
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;
  dim3 grid(1, num_blocks);
  dim3 threads(16, 16);


  FILE *fpout = fopen("backprop_skylake_vector_times.txt","a+");
  if(!fpout) {
    printf("Error Saving stats\n");
    exit(1);
  }


  // if file is empty then give it headers
  fseek (fpout, 0, SEEK_END); //move pointer to the end of file
  if ( ftell(fpout) == 0 ) { // if the pos is 0, it is empty
    fprintf(fpout,"%-10s, %-10s, %-10s, ", "Affinity", "Usage", "#elements");

    // application specific
    //fprintf(fpout,"rows, cols, height, ");

    fprintf(fpout,"%-20s\n", "time");
  }
  fseek(fpout, 0, SEEK_SET); // reset pointer to start of file


  input_weights_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));

  // this preprocessing stage is added to correct the bugs of wrong memcopy
  // using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
      m++;
    }
  }

//  phiMalloc(&input_cuda, (in + 1));
//  phiMalloc(&input_hidden_cuda, (in + 1) * (hid + 1));
//  phiMalloc(&output_hidden_cuda, (hid + 1));
//  phiMalloc(&hidden_partial_sum, num_blocks * WIDTH);

  input_cuda = (float *)malloc((in+1)*sizeof(float));
  input_hidden_cuda = (float *)malloc((in+1)*(hid+1)*sizeof(float));
  output_hidden_cuda = (float *)malloc((hid+1)*sizeof(float));
  hidden_partial_sum = (float *)malloc(num_blocks*WIDTH*sizeof(float));

  printf("Performing Xeon Phi computation\n");

  // printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
//  phiMemcpy(input_cuda, net->input_units, (in + 1), CpuToPhi);
//  phiMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1), CpuToPhi);

  memcpy(input_cuda, net->input_units, (in+1)*sizeof(float));
  memcpy(input_hidden_cuda, input_weights_one_dim, (in+1)*(hid+1)*sizeof(float));

  // cpu kernel 1
  //bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
  double total_time=0.0;
  double start_time = gettime_ms();

  //printf("# threads: %d\n", threads);

  bpnn_layerforward_CUDA( affinity, usage, grid, threads, input_cuda, output_hidden_cuda, input_hidden_cuda, hidden_partial_sum, in, hid);

  //printf("checkpoint2\n");

  //bpnn_layerforward_CUDA( affinity, usage, grid, threads, net->input_units, output_hidden_cuda, input_weights_one_dim, hidden_partial_sum, in, hid);

  double end_time = gettime_ms();
  total_time+=end_time-start_time;

  // TODO: barrier?
  //cudaThreadSynchronize();

  // TODO: error checking
  //cudaError_t error = cudaGetLastError();
  //if (error != cudaSuccess) {
  //    printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
  //    exit(EXIT_FAILURE);
  //}

//  phiMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH, PhiToCpu);
  memcpy(partial_sum, hidden_partial_sum, num_blocks*WIDTH*sizeof(float));

  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j - 1];
    }
    sum += net->input_weights[0][j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }

///////////////

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  // cpu kernel 2
  //bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

///////////////


  //phiMalloc(&hidden_delta_cuda, (hid + 1));
  //phiMalloc(&input_prev_weights_cuda, (in + 1) * (hid + 1));

  //phiMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1), CpuToPhi);

  hidden_delta_cuda = net->hidden_delta;
  input_prev_weights_cuda = input_weights_prev_one_dim;
  input_hidden_cuda = input_weights_one_dim;

  //phiMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1), CpuToPhi);
  //phiMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1), CpuToPhi);

//  printf("\nExecution on Xeon Phi begins!\n");
//	time = gettime_ms();

  start_time = gettime_ms();
//  bpnn_adjust_weights_cuda(affinity, usage, grid, threads, hidden_delta_cuda, hid, input_cuda, in, input_hidden_cuda, input_prev_weights_cuda);
  bpnn_adjust_weights_cuda(affinity, usage, grid, threads, net->hidden_delta, hid, net->input_units, in, input_weights_one_dim, input_weights_prev_one_dim);
  end_time = gettime_ms();
  total_time+=end_time-start_time;

  //printf("Compute time: %lf\n", total_time);
  fprintf(fpout,"%-10d, %-10lf, %-10d, ", affinity, usage, layer_size);
  //fprintf(fpout,"%d ", no_of_nodes);
  fprintf(fpout,"%-20lf\n", total_time);
  fclose(fpout);

  //phiMemcpy(net->input_units, input_cuda, (in + 1), PhiToCpu);
  //phiMemcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1), PhiToCpu);


  //phiFree(input_cuda);
  //phiFree(output_hidden_cuda);
  //phiFree(input_hidden_cuda);
  //phiFree(hidden_partial_sum);
  //phiFree(input_prev_weights_cuda);
  //phiFree(hidden_delta_cuda);
  
  free(output_hidden_cuda);
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void load(BPNN *net) {
  float *units;
  //int nr, nc, imgsize, 
  int nr, i, j, k;

  nr = layer_size;
  
  //imgsize = nr * nc;
  units = net->input_units;

  k = 1;
  for (i = 0; i < nr; i++) {
    units[k] = (float) rand()/RAND_MAX ;
	  k++;
  }
}

void backprop_face() {
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);

  //bpnn_print(net);

  bpnn_free(net);
  printf("Training done\n");
}


void run(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: backprop <affinity> <usage> <num of input elements>\n");
    exit(1);
  }

  affinity = atoi(argv[1]);
  usage = atof(argv[2]);

  layer_size = atoi(argv[3]);
  if (layer_size % 16 != 0) {
    fprintf(stderr, "The number of input points must be divided by 16\n");
    exit(1);
  }

  int seed = 7;
  bpnn_initialize(seed);
  backprop_face();
}

int main( int argc, char** argv) {
  phiSetDevice(1);
  run(argc, argv);

  return EXIT_SUCCESS;
}

