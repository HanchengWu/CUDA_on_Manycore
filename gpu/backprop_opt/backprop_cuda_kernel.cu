#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"


__global__ void bpnn_layerforward_CUDA(float *input_cuda,
                                       float *output_hidden_cuda,
                                       float *input_hidden_cuda,
                                       float *hidden_partial_sum, int in,
                                       int hid) {

  int by = blockIdx.y;
  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT][WIDTH];

  __shared__ int index_save[256];

  for ( int tid = threadIdx.x; tid < 256; tid+=16 ) {
    int tx = tid/16;
    int ty = tid%16;

    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    index_save[tid] = index;

    int index_in = HEIGHT * by + ty + 1;

    if (tx == 0)
        input_node[ty] = input_cuda[index_in];
  }

  //__syncthreads();

  for ( int tid = threadIdx.x; tid < 256; tid+=16 ) {
    int tx = tid/16;
    int ty = tid%16;
    int index_recovered = index_save[tid];

    weight_matrix[ty][tx] = input_hidden_cuda[index_recovered];
  }

  //__syncthreads();

  for ( int tid = threadIdx.x; tid < 256; tid+=16 ) {
    int tx = tid/16;
    int ty = tid%16;
    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];
  }

  //__syncthreads();

  for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
    for ( int tid = threadIdx.x; tid < 256; tid+=16 ) {
      int tx = tid/16;
      int ty = tid%16;

      if (ty % power_two == 0)
          weight_matrix[ty][tx] =
              weight_matrix[ty][tx] + weight_matrix[ty + power_two / 2][tx];

    }
    //  __syncthreads();
  }

  for ( int tid = threadIdx.x; tid < 256; tid+=16 ) {
    int tx = tid/16;
    int ty = tid%16;

    int index_recovered = index_save[tid];

    input_hidden_cuda[index_recovered] = weight_matrix[ty][tx];
  }

  //__syncthreads();

  for ( int tid = threadIdx.x; tid < 256; tid+=16 ) {
    int tx = tid/16;
    int ty = tid%16;

    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
    }
  }
}

__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
                                         int in, float *w, float *oldw) {

    int by = blockIdx.y;
    int index_x_save[256];

    for ( int tid = threadIdx.x; tid < 256; tid+=16 ) {
      int tx = tid/16;
      int ty = tid%16;

      int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);

      int index_y = HEIGHT * by + ty + 1;
      int index_x = tx + 1;
      index_x_save[tid] = index_x;
      // eta = 0.3;
      // momentum = 0.3;
  
      w[index] +=
          ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      oldw[index] =
          ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      /** TEMP REMOVE **/
      //printf("%lf %lf %lf %lf %lf %lf\n", ETA, delta[index_x], ly[index_y], MOMENTUM, w[index], oldw[index]);
    }

    //__syncthreads();

    for ( int tid = threadIdx.x; tid < 256; tid+=16 ) {
      //int tx = tid/16;
      int ty = tid%16;

      int index_x_recovered = index_x_save[tid];

      if (ty == 0 && by == 0) {
          w[index_x_recovered] += ((ETA * delta[index_x_recovered]) + (MOMENTUM * oldw[index_x_recovered]));
          oldw[index_x_recovered] = ((ETA * delta[index_x_recovered]) + (MOMENTUM * oldw[index_x_recovered]));
      }
    }
}
#endif
