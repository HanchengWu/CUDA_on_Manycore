#ifndef GENERAL_KERNELS_CUH_
#define GENERAL_KERNELS_CUH_

#include "general_stdinc.h"
#include "main.h"

#include "topology_function.h"

#include "cudaCheckError.h"


__global__ void fixed_topology_kernel(  	unsigned* result_bit_vector,
											char* stream_sequences,
											#ifdef RODC_ON
											struct preprocessed_full_reference_char_sequence const * __restrict__  preprocessed_input,
											#else
											struct preprocessed_full_reference_char_sequence * preprocessed_input,
											#endif
											unsigned bit_chunks_per_state_vector,
											unsigned char_filled_ints_per_packet,
											unsigned num_packets, //cf.packets_per_kernel_launch is used later in reading and analyzing the results
											unsigned warp_efficient_stream_count,
											unsigned occupancy_efficient_stream_count,
											unsigned ref_block_count,
											unsigned batch_count,	//the number of word-chunks (32-bit) that holds the symbols of NFA s on each block
											unsigned accepting_states_count  );

__device__ unsigned match_check(char s_char,unsigned passed_chars/*unsigned offset_in_soa*/,unsigned tx_char_count,
								#ifdef RODC_ON
								struct preprocessed_full_reference_char_sequence const * __restrict__ preprocessed_input,
								#else
								struct preprocessed_full_reference_char_sequence * preprocessed_input,
								#endif
								unsigned ref_block_count,
								unsigned batch_count);

__device__ void character_transitions_update(unsigned * cfBV, unsigned ctof,unsigned src, unsigned dst, unsigned mask, unsigned has_wildcard,unsigned has_neg,unsigned has_positive);

__device__ void update_StateVector(unsigned * cfBV, unsigned ctof, unsigned bit_chunks_per_state_vector, unsigned accepting_states_count);

__device__ void fill_results_array(unsigned * result_bit_vector, unsigned * cfBV, unsigned ctof, unsigned bit_chunks_per_state_vector, unsigned accepting_states_count, unsigned packet_idx, unsigned * p_idx);
#endif
