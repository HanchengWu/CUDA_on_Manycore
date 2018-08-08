//#include "stdinc.h"
#include "general_kernels.h"

__global__ void fixed_topology_kernel(  	unsigned* result_bit_vector,
											char* stream_sequences,
											#ifdef RODC_ON
											struct preprocessed_full_reference_char_sequence const * __restrict__  preprocessed_input,
											#else
											struct preprocessed_full_reference_char_sequence * preprocessed_input,
											#endif
											unsigned bit_chunks_per_state_vector, //number of integers in sv
											unsigned char_filled_ints_per_packet,
											unsigned num_packets, //cf.packets_per_kernel_launch is used later in reading and analyzing the results
											unsigned warp_efficient_stream_count,
											unsigned occupancy_efficient_stream_count,
											unsigned ref_block_count,
											unsigned batch_count,	//the number of word-chunks (32-bit) that holds the symbols of NFA s on each block
											unsigned accepting_states_count  ){

	//Registers
	unsigned ctof =							bit_chunks_per_state_vector;//will be used for current versus future offset
	unsigned * u_stream_sequences = 	  	(unsigned *)stream_sequences;
	unsigned packet_idx = 					0;	//sequence index in streaming sequences
	unsigned p_idx =						0;	//preprocessed sequence index
	unsigned char_filled_int_idx =			0;	//intra-block sequence index
	char 	s_char =						0;	//single decompressed character from the stream, used when handling one character at a time
	unsigned s_chars = 						0;	//MAX_K compressed characters from stream, where MAX_K is the maximum number of characters that will fit in the bit pattern selected via the libKmer.a library
	int s_char_index = 						0;	//compressed char index, index 0 to MAX_K-1 that determines which character within a Compressed_kmer structure is being targeted

	//id of the input stream based on the thread and block count
	int stream_id = warp_efficient_stream_count*(blockIdx.x/ref_block_count) + (threadIdx.x/batch_count);
	if (stream_id > (occupancy_efficient_stream_count*warp_efficient_stream_count)-1 ) printf("stream-id has exceed the allowed range!\n");/*stream_id = 0;*///for the last idle threads that might exceed the allowed stream numbers

	//Shared Memory
	extern __shared__ unsigned cfBV[];//Both current and future bit vectors, current first half, future second

	for(packet_idx=0; packet_idx < char_filled_ints_per_packet*num_packets; packet_idx+=char_filled_ints_per_packet){
		//set first state always 1 (i.e. non-anchored start state) M: the traversal resets on each packet
    	cfBV[threadIdx.x] = 0xFFFFFFFF;
	    cfBV[ctof + threadIdx.x] = 0xFFFFFFFF;

		for(char_filled_int_idx = packet_idx; char_filled_int_idx < packet_idx + char_filled_ints_per_packet; char_filled_int_idx += 1){ //char_filled_int_idx is the counter on a single stream

			s_chars = u_stream_sequences[(stream_id*char_filled_ints_per_packet*num_packets)+char_filled_int_idx];//reference stream offset + counter in the stream

			for(s_char_index = 3; s_char_index >= 0; s_char_index -= 1){
////////////////////start "Clock Cycle"/////////////////////////////////////////////////////////////////////////////
				s_char = (char)(  (s_chars>>(8*(3-s_char_index))) & 0x000000FFu   );//threads in a block get the same copy of s_char (this is one character at a time paradigm)

//				  if( packet_idx==0 && (char_filled_int_idx == 0) || ((char_filled_int_idx == 1)  && (s_char_index ==  3 /*|| s_char_index == 2 || s_char_index == 1 || s_char_index == 0 */)) ){
//					if((threadIdx.x ==0) && (blockIdx.x==0) ) printf("input= %08x , s_char_index=%d \n",s_char,s_char_index);//printf("input= %c\n",s_char);
				topology_specific_traversal(s_char,preprocessed_input,cfBV,ctof,bit_chunks_per_state_vector,accepting_states_count,ref_block_count,batch_count);

////////////////////end "Clock Cycle"///////////////////////////////////////////////////////////////////////////////////

//				}//remove since its for debug

			}//Stream 1 char loop

		}//Stream 16 char loop

		//Write to global memory  here
		fill_results_array(result_bit_vector, cfBV, ctof, bit_chunks_per_state_vector, accepting_states_count , packet_idx, &p_idx);

	}
}


__device__ unsigned match_check(char s_char,
								//unsigned offset_in_SOA, //is calculated by the compiler and depends on the src and dst of the TX
								unsigned passed_chars, //is calculated by the compiler and depends on the src and dst of the TX
								unsigned tx_char_count,
								#ifdef RODC_ON
								struct preprocessed_full_reference_char_sequence const * __restrict__ preprocessed_input,
								#else
								struct preprocessed_full_reference_char_sequence * preprocessed_input,
								#endif
								unsigned ref_block_count,
								unsigned batch_count){//n is the number of 32-bit words that contain all N nfas
	unsigned mask_res  = 	0x00000000;

	int tx_char_iterator;
	for(tx_char_iterator = 0; tx_char_iterator < tx_char_count; tx_char_iterator++){

		unsigned mask  = 	0x00000000;
		unsigned offset_in_SOA = passed_chars*batch_count;

		unsigned p_chars_7 = 	preprocessed_input[(blockIdx.x%ref_block_count)].SOA_chunks_7_word[offset_in_SOA+(batch_count*tx_char_iterator)+(threadIdx.x%batch_count)];
		unsigned p_chars_6 =	preprocessed_input[(blockIdx.x%ref_block_count)].SOA_chunks_6_word[offset_in_SOA+(batch_count*tx_char_iterator)+(threadIdx.x%batch_count)];
		unsigned p_chars_5 =	preprocessed_input[(blockIdx.x%ref_block_count)].SOA_chunks_5_word[offset_in_SOA+(batch_count*tx_char_iterator)+(threadIdx.x%batch_count)];
		unsigned p_chars_4 =	preprocessed_input[(blockIdx.x%ref_block_count)].SOA_chunks_4_word[offset_in_SOA+(batch_count*tx_char_iterator)+(threadIdx.x%batch_count)];
		unsigned p_chars_3 =	preprocessed_input[(blockIdx.x%ref_block_count)].SOA_chunks_3_word[offset_in_SOA+(batch_count*tx_char_iterator)+(threadIdx.x%batch_count)];
		unsigned p_chars_2 =	preprocessed_input[(blockIdx.x%ref_block_count)].SOA_chunks_2_word[offset_in_SOA+(batch_count*tx_char_iterator)+(threadIdx.x%batch_count)];
		unsigned p_chars_1 =	preprocessed_input[(blockIdx.x%ref_block_count)].SOA_chunks_1_word[offset_in_SOA+(batch_count*tx_char_iterator)+(threadIdx.x%batch_count)];
		unsigned p_chars_0 =	preprocessed_input[(blockIdx.x%ref_block_count)].SOA_chunks_0_word[offset_in_SOA+(batch_count*tx_char_iterator)+(threadIdx.x%batch_count)];

/*		if((threadIdx.x ==31) && (blockIdx.x==0) )
			printf("pchar7= %01x,pchar6= %01x,pchar5= %01x,pchar4= %01x,pchar3= %01x,pchar2= %01x,pchar1= %01x,pchar0= %01x /ofset= %d\n", p_chars_7,p_chars_6,p_chars_5,p_chars_4,p_chars_3,p_chars_2,p_chars_1,p_chars_0,offset_in_SOA );
*/

		int j = 0;
		char p_char;

		for(j=3; j>=0; j-=1){
			p_char = (char)((p_chars_7>>(8*(3-j))) & 0x000000FFu);
			if(s_char==p_char){
				mask |= ( (0x1) << j );
			}
		}
		mask = mask<<4;
		for(j=3; j>=0; j-=1){
			p_char = (char)((p_chars_6>>(8*(3-j))) & 0x000000FFu);
			if(s_char==p_char){
				mask |= ( (0x1) << j );
			}
		}
		mask = mask<<4;
		for(j=3; j>=0; j-=1){
			p_char = (char)((p_chars_5>>(8*(3-j))) & 0x000000FFu);
			if(s_char==p_char){
				mask |= ( (0x1) << j );
			}
		}
		mask = mask<<4;
		for(j=3; j>=0; j-=1){
			p_char = (char)((p_chars_4>>(8*(3-j))) & 0x000000FFu);
			if(s_char==p_char){
				mask |= ( (0x1) << j );
			}
		}
		mask = mask<<4;
		for(j=3; j>=0; j-=1){
			p_char = (char)((p_chars_3>>(8*(3-j))) & 0x000000FFu);
			if(s_char==p_char){
				mask |= ( (0x1) << j );
			}
		}
		mask = mask<<4;
		for(j=3; j>=0; j-=1){
			p_char = (char)((p_chars_2>>(8*(3-j))) & 0x000000FFu);
			if(s_char==p_char){
				mask |= ( (0x1) << j );
			}
		}
		mask = mask<<4;
		for(j=3; j>=0; j-=1){
			p_char = (char)((p_chars_1>>(8*(3-j))) & 0x000000FFu);
			if(s_char==p_char){
				mask |= ( (0x1) << j );
			}
		}
		mask = mask<<4;
		for(j=3; j>=0; j-=1){
			p_char = (char)((p_chars_0>>(8*(3-j))) & 0x000000FFu);
			if(s_char==p_char){
				mask |= ( (0x1) << j );
			}
		}
		mask_res |= mask;
	}
/*	if((threadIdx.x ==31) && (blockIdx.x==0) ) printf("mask= %08x , ofset= %d\n",mask_res,offset_in_SOA );*/
	return mask_res;
}

__device__ void character_transitions_update(unsigned * cfBV, unsigned ctof,unsigned src, unsigned dst, unsigned mask,unsigned has_wildcard,unsigned has_neg,unsigned has_positive){

	unsigned current_index = (blockDim.x*src) + threadIdx.x;
	unsigned future_index = (blockDim.x*dst) + threadIdx.x;

	unsigned current = cfBV[current_index]; //if(((src==8) /*|| (src==4)*/) && (threadIdx.x ==0) && (blockIdx.x==0) ) printf("current[%d]= %08x, ctof=%d \n",src,current,ctof );

	if(has_wildcard){
		cfBV[ctof + future_index] |= current;
	}else if(has_positive){ //if((src==8) && (threadIdx.x ==0) && (blockIdx.x==0) ) printf("before-> current[%d]= %08x, mask= %08x, future[%d]=%08x \n",src,current,mask,dst,cfBV[ctof + future_index] );
		cfBV[ctof + future_index] |= mask & current;
//		if((src==8) && (threadIdx.x ==0) && (blockIdx.x==0) ) printf("after-> current[%d]= %08x, mask= %08x, future[%d]=%08x \n",src,current,mask,dst,cfBV[ctof + future_index] );
	}else if(has_neg){
		cfBV[ctof + future_index] |= ~(mask) & current;
	}
//	if((dst==9) && (threadIdx.x ==0) && (blockIdx.x==0) ) printf("src = %d, current[%d]= %08x,future[9]= %08x \n",src,src,current,cfBV[ctof + future_index] );
}

__device__ void update_StateVector(unsigned * cfBV, unsigned ctof, unsigned bit_chunks_per_state_vector, unsigned accepting_states_count){
	int i=0;
	for(i=threadIdx.x + blockDim.x; i<bit_chunks_per_state_vector-((accepting_states_count)*blockDim.x); i+=blockDim.x){//M: jumping from the initial state and also ignoring the accepting states region
		cfBV[i] = cfBV[ctof + i];
		cfBV[ctof + i] = 0;
	}//update current with future loop
	for(i=threadIdx.x + (bit_chunks_per_state_vector-((accepting_states_count)*blockDim.x)); i<bit_chunks_per_state_vector; i+=blockDim.x){
		cfBV[i] |= cfBV[ctof + i];
		cfBV[ctof + i] = 0;
	}//accumulative update of accepting states
}

__device__ void fill_results_array(unsigned * result_bit_vector, unsigned * cfBV, unsigned ctof, unsigned bit_chunks_per_state_vector, unsigned accepting_states_count, unsigned packet_idx, unsigned * p_idx){//we use packet_idx only in case that we want to debug based on results of the first packet
	int i = 0;
	#ifdef STATE_VECTOR_DEBUG
	if( (packet_idx==0)){ //M: so for debug all the bits of the state vector of the first packet only will be copied to the result bit vector
		//if(blockIdx.x==0)
		for(i=threadIdx.x; i<bit_chunks_per_state_vector; i+=blockDim.x){
			result_bit_vector[bit_chunks_per_state_vector*blockIdx.x + i] = cfBV[i];
			cfBV[i]=0;
		}
	}
	#else //Copy only accepting state vector portions


	unsigned offset = (*p_idx)*(accepting_states_count*gridDim.x*blockDim.x) + ( accepting_states_count*blockDim.x*blockIdx.x );
	for(i=(bit_chunks_per_state_vector - (accepting_states_count*blockDim.x) ) + threadIdx.x; i<bit_chunks_per_state_vector; i+=blockDim.x){ //should result in three 128 byte writes to global mem
		result_bit_vector[ offset + i-((bit_chunks_per_state_vector - (accepting_states_count*blockDim.x) ))] = cfBV[i];	// i-(bla bla) since we only transfer the accepting states portion
	}

	for(i=threadIdx.x /*+blockDim.x*/; i<2*bit_chunks_per_state_vector; i+=blockDim.x){
		cfBV[i] = 0; //Zero out bit vector for next packeti M: initializaion of current SV takes place for next packet in next iteration
		//cfBV[ctof + i] = 0;
	}
	(*p_idx)++;
	#endif
}


