//Marziyeh Nourian
#ifndef GENERAL_STDINC_H_
#define GENERAL_STDINC_H_

//#ifndef RODC_ON
//#define RODC_OFF
//#endif

//#define PACKET_SIZE  1*1000		//(Byte)1 K Byte

//#define STATE_VECTOR_DEBUG

#define MAX_SOA_CHUNCK_COUNT 50000 //what should it be for our largest dataset? (27616 for synthetic 300 states)

//More typical includes below*******************************/
//**********************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
///#include <cuda.h>
#include <algorithm>
///#include <cuda_profiler_api.h>

typedef u_int32_t Compressed_kmer;
struct general_config {

	char**		stream_sequence_filename/*[200]*/;

	unsigned	trace_num;

	char		compiler_output_filename[200];

	char		application[20]; 			//used for data collection purpose

	unsigned	packets_per_kernel_launch;	//the only usage is that upon start of each packet, the shared memory will be wipped and results will be stored for each packet

	unsigned	packet_size;			//must be a factor of sizeof(unsigned) because we calculate this by deviding packet size and sizeof(unsigned)

	unsigned 	char_filled_ints_per_packet;

	unsigned 	gpu_device;

	unsigned	ref_block_count;

	unsigned	batch_count;

	unsigned	batch_size;

	unsigned 	threads_per_block;

	unsigned 	warp_efficient_stream_count;

	unsigned	occupancy_efficient_stream_count;

	unsigned	nfa_size;

	unsigned	accepting_states_count;

	unsigned	SOA_chunk_count;

	unsigned	streams_count;

	unsigned 	blocks_count;

	unsigned	bit_chunks_per_state_vector;

	

//#ifdef POSTPROC
	unsigned 	k;//kmer length
	unsigned	d;//distance
//#endif

	/*host and device input streams*/
	char * fc_streaming_sequences_d;
	char * fc_streaming_sequences_h;

	/*host and device preprocessed symbols*/
	struct preprocessed_full_reference_char_sequence * fc_preprocessed_input_d;
	struct preprocessed_full_reference_char_sequence * fc_preprocessed_input_h;

	/*host and device bit vectors for results*/
	unsigned * result_bit_vector_d;
	unsigned * result_bit_vector_h;


	FILE * final_test_outfile;
	char start_stamp[100];
	double start;
	double stop;
	double preprocessing;
	double stream_to_dev;
	double kernel;
	double result_from_dev;
	double post_processing;
};

extern struct general_config cf0;

__inline__ double gettime() {
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
}



#endif // GENERAL_STDINC_H_
