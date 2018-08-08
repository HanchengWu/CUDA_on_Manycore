
#include "topology_function.h"


__device__ void topology_specific_traversal(char s_char,
					#ifdef RODC_ON
					struct preprocessed_full_reference_char_sequence const * __restrict__  preprocessed_input,
					#else
					struct preprocessed_full_reference_char_sequence * preprocessed_input,
					#endif
					unsigned * cfBV,
					unsigned ctof,
					unsigned bit_chunks_per_state_vector,
					unsigned accepting_states_count,
					unsigned ref_block_count,
					unsigned batch_count){
	unsigned src_state,mask;

#ifdef	Levenshtein_k8_d1_kernel
	for(src_state = 0; src_state < 18; src_state++){
 if (src_state == 0){
			/***dest 1***/
			character_transitions_update(cfBV, ctof,0,1,mask,1,0,0);
			/***dest 2***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,0,1);
			/***dest 3***/
			character_transitions_update(cfBV, ctof,0,3,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*3) + threadIdx.x] |= cfBV[ctof + (blockDim.x*0) + threadIdx.x];
		}else if (src_state == 1){
			/***dest 3***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,3,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 3***/
			character_transitions_update(cfBV, ctof,2,3,mask,1,0,0);
			/***dest 4***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,4,mask,0,0,1);
			/***dest 5***/
			character_transitions_update(cfBV, ctof,2,5,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*5) + threadIdx.x] |= cfBV[ctof + (blockDim.x*2) + threadIdx.x];
		}else if (src_state == 3){
			/***dest 5***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,5,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 5***/
			character_transitions_update(cfBV, ctof,4,5,mask,1,0,0);
			/***dest 6***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,6,mask,0,0,1);
			/***dest 7***/
			character_transitions_update(cfBV, ctof,4,7,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*7) + threadIdx.x] |= cfBV[ctof + (blockDim.x*4) + threadIdx.x];
		}else if (src_state == 5){
			/***dest 7***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,7,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 7***/
			character_transitions_update(cfBV, ctof,6,7,mask,1,0,0);
			/***dest 8***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,8,mask,0,0,1);
			/***dest 9***/
			character_transitions_update(cfBV, ctof,6,9,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*9) + threadIdx.x] |= cfBV[ctof + (blockDim.x*6) + threadIdx.x];
		}else if (src_state == 7){
			/***dest 9***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,9,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 9***/
			character_transitions_update(cfBV, ctof,8,9,mask,1,0,0);
			/***dest 10***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,10,mask,0,0,1);
			/***dest 11***/
			character_transitions_update(cfBV, ctof,8,11,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*11) + threadIdx.x] |= cfBV[ctof + (blockDim.x*8) + threadIdx.x];
		}else if (src_state == 9){
			/***dest 11***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,11,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 11***/
			character_transitions_update(cfBV, ctof,10,11,mask,1,0,0);
			/***dest 12***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,12,mask,0,0,1);
			/***dest 13***/
			character_transitions_update(cfBV, ctof,10,13,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*13) + threadIdx.x] |= cfBV[ctof + (blockDim.x*10) + threadIdx.x];
		}else if (src_state == 11){
			/***dest 13***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,13,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 13***/
			character_transitions_update(cfBV, ctof,12,13,mask,1,0,0);
			/***dest 14***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,14,mask,0,0,1);
			/***dest 15***/
			character_transitions_update(cfBV, ctof,12,15,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*15) + threadIdx.x] |= cfBV[ctof + (blockDim.x*12) + threadIdx.x];
		}else if (src_state == 13){
			/***dest 15***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,15,mask,0,0,1);
		}else if (src_state == 14){
			/***dest 15***/
			character_transitions_update(cfBV, ctof,14,15,mask,1,0,0);
			/***dest 16***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,16,mask,0,0,1);
			/***dest 17***/
			character_transitions_update(cfBV, ctof,14,17,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*17) + threadIdx.x] |= cfBV[ctof + (blockDim.x*14) + threadIdx.x];
		}else if (src_state == 15){
			/***dest 17***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,17,mask,0,0,1);
		}else if (src_state == 16){
			/***dest 17***/
			character_transitions_update(cfBV, ctof,16,17,mask,1,0,0);
		}else if (src_state == 17){
		}
	}
#endif

#ifdef levenshtein_k20_d3_kernel
	for(src_state = 0; src_state < 84; src_state++){
 if (src_state == 0){
			/***dest 1***/
			character_transitions_update(cfBV, ctof,0,1,mask,1,0,0);
			/***dest 4***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,4,mask,0,0,1);
			/***dest 5***/
			character_transitions_update(cfBV, ctof,0,5,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*5) + threadIdx.x] |= cfBV[ctof + (blockDim.x*0) + threadIdx.x];
			cfBV[ctof + (blockDim.x*10) + threadIdx.x] |= cfBV[ctof + (blockDim.x*0) + threadIdx.x];
			cfBV[ctof + (blockDim.x*15) + threadIdx.x] |= cfBV[ctof + (blockDim.x*0) + threadIdx.x];
		}else if (src_state == 1){
			/***dest 2***/
			character_transitions_update(cfBV, ctof,1,2,mask,1,0,0);
			/***dest 5***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,5,mask,0,0,1);
			/***dest 6***/
			character_transitions_update(cfBV, ctof,1,6,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*6) + threadIdx.x] |= cfBV[ctof + (blockDim.x*1) + threadIdx.x];
			cfBV[ctof + (blockDim.x*11) + threadIdx.x] |= cfBV[ctof + (blockDim.x*1) + threadIdx.x];
		}else if (src_state == 2){
			/***dest 3***/
			character_transitions_update(cfBV, ctof,2,3,mask,1,0,0);
			/***dest 6***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,6,mask,0,0,1);
			/***dest 7***/
			character_transitions_update(cfBV, ctof,2,7,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*7) + threadIdx.x] |= cfBV[ctof + (blockDim.x*2) + threadIdx.x];
		}else if (src_state == 3){
			/***dest 7***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,7,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 5***/
			character_transitions_update(cfBV, ctof,4,5,mask,1,0,0);
			/***dest 8***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,8,mask,0,0,1);
			/***dest 9***/
			character_transitions_update(cfBV, ctof,4,9,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*9) + threadIdx.x] |= cfBV[ctof + (blockDim.x*4) + threadIdx.x];
			cfBV[ctof + (blockDim.x*14) + threadIdx.x] |= cfBV[ctof + (blockDim.x*4) + threadIdx.x];
			cfBV[ctof + (blockDim.x*19) + threadIdx.x] |= cfBV[ctof + (blockDim.x*4) + threadIdx.x];
		}else if (src_state == 5){
			/***dest 6***/
			character_transitions_update(cfBV, ctof,5,6,mask,1,0,0);
			/***dest 9***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,9,mask,0,0,1);
			/***dest 10***/
			character_transitions_update(cfBV, ctof,5,10,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*10) + threadIdx.x] |= cfBV[ctof + (blockDim.x*5) + threadIdx.x];
			cfBV[ctof + (blockDim.x*15) + threadIdx.x] |= cfBV[ctof + (blockDim.x*5) + threadIdx.x];
		}else if (src_state == 6){
			/***dest 7***/
			character_transitions_update(cfBV, ctof,6,7,mask,1,0,0);
			/***dest 10***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,10,mask,0,0,1);
			/***dest 11***/
			character_transitions_update(cfBV, ctof,6,11,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*11) + threadIdx.x] |= cfBV[ctof + (blockDim.x*6) + threadIdx.x];
		}else if (src_state == 7){
			/***dest 11***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,11,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 9***/
			character_transitions_update(cfBV, ctof,8,9,mask,1,0,0);
			/***dest 12***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,12,mask,0,0,1);
			/***dest 13***/
			character_transitions_update(cfBV, ctof,8,13,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*13) + threadIdx.x] |= cfBV[ctof + (blockDim.x*8) + threadIdx.x];
			cfBV[ctof + (blockDim.x*18) + threadIdx.x] |= cfBV[ctof + (blockDim.x*8) + threadIdx.x];
			cfBV[ctof + (blockDim.x*23) + threadIdx.x] |= cfBV[ctof + (blockDim.x*8) + threadIdx.x];
		}else if (src_state == 9){
			/***dest 10***/
			character_transitions_update(cfBV, ctof,9,10,mask,1,0,0);
			/***dest 13***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,13,mask,0,0,1);
			/***dest 14***/
			character_transitions_update(cfBV, ctof,9,14,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*14) + threadIdx.x] |= cfBV[ctof + (blockDim.x*9) + threadIdx.x];
			cfBV[ctof + (blockDim.x*19) + threadIdx.x] |= cfBV[ctof + (blockDim.x*9) + threadIdx.x];
		}else if (src_state == 10){
			/***dest 11***/
			character_transitions_update(cfBV, ctof,10,11,mask,1,0,0);
			/***dest 14***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,14,mask,0,0,1);
			/***dest 15***/
			character_transitions_update(cfBV, ctof,10,15,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*15) + threadIdx.x] |= cfBV[ctof + (blockDim.x*10) + threadIdx.x];
		}else if (src_state == 11){
			/***dest 15***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,15,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 13***/
			character_transitions_update(cfBV, ctof,12,13,mask,1,0,0);
			/***dest 16***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,16,mask,0,0,1);
			/***dest 17***/
			character_transitions_update(cfBV, ctof,12,17,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*17) + threadIdx.x] |= cfBV[ctof + (blockDim.x*12) + threadIdx.x];
			cfBV[ctof + (blockDim.x*22) + threadIdx.x] |= cfBV[ctof + (blockDim.x*12) + threadIdx.x];
			cfBV[ctof + (blockDim.x*27) + threadIdx.x] |= cfBV[ctof + (blockDim.x*12) + threadIdx.x];
		}else if (src_state == 13){
			/***dest 14***/
			character_transitions_update(cfBV, ctof,13,14,mask,1,0,0);
			/***dest 17***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,17,mask,0,0,1);
			/***dest 18***/
			character_transitions_update(cfBV, ctof,13,18,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*18) + threadIdx.x] |= cfBV[ctof + (blockDim.x*13) + threadIdx.x];
			cfBV[ctof + (blockDim.x*23) + threadIdx.x] |= cfBV[ctof + (blockDim.x*13) + threadIdx.x];
		}else if (src_state == 14){
			/***dest 15***/
			character_transitions_update(cfBV, ctof,14,15,mask,1,0,0);
			/***dest 18***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,18,mask,0,0,1);
			/***dest 19***/
			character_transitions_update(cfBV, ctof,14,19,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*19) + threadIdx.x] |= cfBV[ctof + (blockDim.x*14) + threadIdx.x];
		}else if (src_state == 15){
			/***dest 19***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,19,mask,0,0,1);
		}else if (src_state == 16){
			/***dest 17***/
			character_transitions_update(cfBV, ctof,16,17,mask,1,0,0);
			/***dest 20***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,20,mask,0,0,1);
			/***dest 21***/
			character_transitions_update(cfBV, ctof,16,21,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*21) + threadIdx.x] |= cfBV[ctof + (blockDim.x*16) + threadIdx.x];
			cfBV[ctof + (blockDim.x*26) + threadIdx.x] |= cfBV[ctof + (blockDim.x*16) + threadIdx.x];
			cfBV[ctof + (blockDim.x*31) + threadIdx.x] |= cfBV[ctof + (blockDim.x*16) + threadIdx.x];
		}else if (src_state == 17){
			/***dest 18***/
			character_transitions_update(cfBV, ctof,17,18,mask,1,0,0);
			/***dest 21***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,21,mask,0,0,1);
			/***dest 22***/
			character_transitions_update(cfBV, ctof,17,22,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*22) + threadIdx.x] |= cfBV[ctof + (blockDim.x*17) + threadIdx.x];
			cfBV[ctof + (blockDim.x*27) + threadIdx.x] |= cfBV[ctof + (blockDim.x*17) + threadIdx.x];
		}else if (src_state == 18){
			/***dest 19***/
			character_transitions_update(cfBV, ctof,18,19,mask,1,0,0);
			/***dest 22***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,22,mask,0,0,1);
			/***dest 23***/
			character_transitions_update(cfBV, ctof,18,23,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*23) + threadIdx.x] |= cfBV[ctof + (blockDim.x*18) + threadIdx.x];
		}else if (src_state == 19){
			/***dest 23***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,23,mask,0,0,1);
		}else if (src_state == 20){
			/***dest 21***/
			character_transitions_update(cfBV, ctof,20,21,mask,1,0,0);
			/***dest 24***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,24,mask,0,0,1);
			/***dest 25***/
			character_transitions_update(cfBV, ctof,20,25,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*25) + threadIdx.x] |= cfBV[ctof + (blockDim.x*20) + threadIdx.x];
			cfBV[ctof + (blockDim.x*30) + threadIdx.x] |= cfBV[ctof + (blockDim.x*20) + threadIdx.x];
			cfBV[ctof + (blockDim.x*35) + threadIdx.x] |= cfBV[ctof + (blockDim.x*20) + threadIdx.x];
		}else if (src_state == 21){
			/***dest 22***/
			character_transitions_update(cfBV, ctof,21,22,mask,1,0,0);
			/***dest 25***/
			mask = match_check(s_char,21,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,21,25,mask,0,0,1);
			/***dest 26***/
			character_transitions_update(cfBV, ctof,21,26,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*26) + threadIdx.x] |= cfBV[ctof + (blockDim.x*21) + threadIdx.x];
			cfBV[ctof + (blockDim.x*31) + threadIdx.x] |= cfBV[ctof + (blockDim.x*21) + threadIdx.x];
		}else if (src_state == 22){
			/***dest 23***/
			character_transitions_update(cfBV, ctof,22,23,mask,1,0,0);
			/***dest 26***/
			mask = match_check(s_char,22,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,26,mask,0,0,1);
			/***dest 27***/
			character_transitions_update(cfBV, ctof,22,27,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*27) + threadIdx.x] |= cfBV[ctof + (blockDim.x*22) + threadIdx.x];
		}else if (src_state == 23){
			/***dest 27***/
			mask = match_check(s_char,23,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,23,27,mask,0,0,1);
		}else if (src_state == 24){
			/***dest 25***/
			character_transitions_update(cfBV, ctof,24,25,mask,1,0,0);
			/***dest 28***/
			mask = match_check(s_char,24,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,28,mask,0,0,1);
			/***dest 29***/
			character_transitions_update(cfBV, ctof,24,29,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*29) + threadIdx.x] |= cfBV[ctof + (blockDim.x*24) + threadIdx.x];
			cfBV[ctof + (blockDim.x*34) + threadIdx.x] |= cfBV[ctof + (blockDim.x*24) + threadIdx.x];
			cfBV[ctof + (blockDim.x*39) + threadIdx.x] |= cfBV[ctof + (blockDim.x*24) + threadIdx.x];
		}else if (src_state == 25){
			/***dest 26***/
			character_transitions_update(cfBV, ctof,25,26,mask,1,0,0);
			/***dest 29***/
			mask = match_check(s_char,25,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,25,29,mask,0,0,1);
			/***dest 30***/
			character_transitions_update(cfBV, ctof,25,30,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*30) + threadIdx.x] |= cfBV[ctof + (blockDim.x*25) + threadIdx.x];
			cfBV[ctof + (blockDim.x*35) + threadIdx.x] |= cfBV[ctof + (blockDim.x*25) + threadIdx.x];
		}else if (src_state == 26){
			/***dest 27***/
			character_transitions_update(cfBV, ctof,26,27,mask,1,0,0);
			/***dest 30***/
			mask = match_check(s_char,26,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,30,mask,0,0,1);
			/***dest 31***/
			character_transitions_update(cfBV, ctof,26,31,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*31) + threadIdx.x] |= cfBV[ctof + (blockDim.x*26) + threadIdx.x];
		}else if (src_state == 27){
			/***dest 31***/
			mask = match_check(s_char,27,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,27,31,mask,0,0,1);
		}else if (src_state == 28){
			/***dest 29***/
			character_transitions_update(cfBV, ctof,28,29,mask,1,0,0);
			/***dest 32***/
			mask = match_check(s_char,28,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,32,mask,0,0,1);
			/***dest 33***/
			character_transitions_update(cfBV, ctof,28,33,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*33) + threadIdx.x] |= cfBV[ctof + (blockDim.x*28) + threadIdx.x];
			cfBV[ctof + (blockDim.x*38) + threadIdx.x] |= cfBV[ctof + (blockDim.x*28) + threadIdx.x];
			cfBV[ctof + (blockDim.x*43) + threadIdx.x] |= cfBV[ctof + (blockDim.x*28) + threadIdx.x];
		}else if (src_state == 29){
			/***dest 30***/
			character_transitions_update(cfBV, ctof,29,30,mask,1,0,0);
			/***dest 33***/
			mask = match_check(s_char,29,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,33,mask,0,0,1);
			/***dest 34***/
			character_transitions_update(cfBV, ctof,29,34,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*34) + threadIdx.x] |= cfBV[ctof + (blockDim.x*29) + threadIdx.x];
			cfBV[ctof + (blockDim.x*39) + threadIdx.x] |= cfBV[ctof + (blockDim.x*29) + threadIdx.x];
		}else if (src_state == 30){
			/***dest 31***/
			character_transitions_update(cfBV, ctof,30,31,mask,1,0,0);
			/***dest 34***/
			mask = match_check(s_char,30,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,34,mask,0,0,1);
			/***dest 35***/
			character_transitions_update(cfBV, ctof,30,35,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*35) + threadIdx.x] |= cfBV[ctof + (blockDim.x*30) + threadIdx.x];
		}else if (src_state == 31){
			/***dest 35***/
			mask = match_check(s_char,31,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,35,mask,0,0,1);
		}else if (src_state == 32){
			/***dest 33***/
			character_transitions_update(cfBV, ctof,32,33,mask,1,0,0);
			/***dest 36***/
			mask = match_check(s_char,32,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,32,36,mask,0,0,1);
			/***dest 37***/
			character_transitions_update(cfBV, ctof,32,37,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*37) + threadIdx.x] |= cfBV[ctof + (blockDim.x*32) + threadIdx.x];
			cfBV[ctof + (blockDim.x*42) + threadIdx.x] |= cfBV[ctof + (blockDim.x*32) + threadIdx.x];
			cfBV[ctof + (blockDim.x*47) + threadIdx.x] |= cfBV[ctof + (blockDim.x*32) + threadIdx.x];
		}else if (src_state == 33){
			/***dest 34***/
			character_transitions_update(cfBV, ctof,33,34,mask,1,0,0);
			/***dest 37***/
			mask = match_check(s_char,33,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,33,37,mask,0,0,1);
			/***dest 38***/
			character_transitions_update(cfBV, ctof,33,38,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*38) + threadIdx.x] |= cfBV[ctof + (blockDim.x*33) + threadIdx.x];
			cfBV[ctof + (blockDim.x*43) + threadIdx.x] |= cfBV[ctof + (blockDim.x*33) + threadIdx.x];
		}else if (src_state == 34){
			/***dest 35***/
			character_transitions_update(cfBV, ctof,34,35,mask,1,0,0);
			/***dest 38***/
			mask = match_check(s_char,34,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,34,38,mask,0,0,1);
			/***dest 39***/
			character_transitions_update(cfBV, ctof,34,39,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*39) + threadIdx.x] |= cfBV[ctof + (blockDim.x*34) + threadIdx.x];
		}else if (src_state == 35){
			/***dest 39***/
			mask = match_check(s_char,35,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,39,mask,0,0,1);
		}else if (src_state == 36){
			/***dest 37***/
			character_transitions_update(cfBV, ctof,36,37,mask,1,0,0);
			/***dest 40***/
			mask = match_check(s_char,36,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,36,40,mask,0,0,1);
			/***dest 41***/
			character_transitions_update(cfBV, ctof,36,41,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*41) + threadIdx.x] |= cfBV[ctof + (blockDim.x*36) + threadIdx.x];
			cfBV[ctof + (blockDim.x*46) + threadIdx.x] |= cfBV[ctof + (blockDim.x*36) + threadIdx.x];
			cfBV[ctof + (blockDim.x*51) + threadIdx.x] |= cfBV[ctof + (blockDim.x*36) + threadIdx.x];
		}else if (src_state == 37){
			/***dest 38***/
			character_transitions_update(cfBV, ctof,37,38,mask,1,0,0);
			/***dest 41***/
			mask = match_check(s_char,37,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,41,mask,0,0,1);
			/***dest 42***/
			character_transitions_update(cfBV, ctof,37,42,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*42) + threadIdx.x] |= cfBV[ctof + (blockDim.x*37) + threadIdx.x];
			cfBV[ctof + (blockDim.x*47) + threadIdx.x] |= cfBV[ctof + (blockDim.x*37) + threadIdx.x];
		}else if (src_state == 38){
			/***dest 39***/
			character_transitions_update(cfBV, ctof,38,39,mask,1,0,0);
			/***dest 42***/
			mask = match_check(s_char,38,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,38,42,mask,0,0,1);
			/***dest 43***/
			character_transitions_update(cfBV, ctof,38,43,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*43) + threadIdx.x] |= cfBV[ctof + (blockDim.x*38) + threadIdx.x];
		}else if (src_state == 39){
			/***dest 43***/
			mask = match_check(s_char,39,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,43,mask,0,0,1);
		}else if (src_state == 40){
			/***dest 41***/
			character_transitions_update(cfBV, ctof,40,41,mask,1,0,0);
			/***dest 44***/
			mask = match_check(s_char,40,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,44,mask,0,0,1);
			/***dest 45***/
			character_transitions_update(cfBV, ctof,40,45,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*45) + threadIdx.x] |= cfBV[ctof + (blockDim.x*40) + threadIdx.x];
			cfBV[ctof + (blockDim.x*50) + threadIdx.x] |= cfBV[ctof + (blockDim.x*40) + threadIdx.x];
			cfBV[ctof + (blockDim.x*55) + threadIdx.x] |= cfBV[ctof + (blockDim.x*40) + threadIdx.x];
		}else if (src_state == 41){
			/***dest 42***/
			character_transitions_update(cfBV, ctof,41,42,mask,1,0,0);
			/***dest 45***/
			mask = match_check(s_char,41,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,45,mask,0,0,1);
			/***dest 46***/
			character_transitions_update(cfBV, ctof,41,46,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*46) + threadIdx.x] |= cfBV[ctof + (blockDim.x*41) + threadIdx.x];
			cfBV[ctof + (blockDim.x*51) + threadIdx.x] |= cfBV[ctof + (blockDim.x*41) + threadIdx.x];
		}else if (src_state == 42){
			/***dest 43***/
			character_transitions_update(cfBV, ctof,42,43,mask,1,0,0);
			/***dest 46***/
			mask = match_check(s_char,42,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,46,mask,0,0,1);
			/***dest 47***/
			character_transitions_update(cfBV, ctof,42,47,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*47) + threadIdx.x] |= cfBV[ctof + (blockDim.x*42) + threadIdx.x];
		}else if (src_state == 43){
			/***dest 47***/
			mask = match_check(s_char,43,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,47,mask,0,0,1);
		}else if (src_state == 44){
			/***dest 45***/
			character_transitions_update(cfBV, ctof,44,45,mask,1,0,0);
			/***dest 48***/
			mask = match_check(s_char,44,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,48,mask,0,0,1);
			/***dest 49***/
			character_transitions_update(cfBV, ctof,44,49,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*49) + threadIdx.x] |= cfBV[ctof + (blockDim.x*44) + threadIdx.x];
			cfBV[ctof + (blockDim.x*54) + threadIdx.x] |= cfBV[ctof + (blockDim.x*44) + threadIdx.x];
			cfBV[ctof + (blockDim.x*59) + threadIdx.x] |= cfBV[ctof + (blockDim.x*44) + threadIdx.x];
		}else if (src_state == 45){
			/***dest 46***/
			character_transitions_update(cfBV, ctof,45,46,mask,1,0,0);
			/***dest 49***/
			mask = match_check(s_char,45,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,49,mask,0,0,1);
			/***dest 50***/
			character_transitions_update(cfBV, ctof,45,50,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*50) + threadIdx.x] |= cfBV[ctof + (blockDim.x*45) + threadIdx.x];
			cfBV[ctof + (blockDim.x*55) + threadIdx.x] |= cfBV[ctof + (blockDim.x*45) + threadIdx.x];
		}else if (src_state == 46){
			/***dest 47***/
			character_transitions_update(cfBV, ctof,46,47,mask,1,0,0);
			/***dest 50***/
			mask = match_check(s_char,46,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,50,mask,0,0,1);
			/***dest 51***/
			character_transitions_update(cfBV, ctof,46,51,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*51) + threadIdx.x] |= cfBV[ctof + (blockDim.x*46) + threadIdx.x];
		}else if (src_state == 47){
			/***dest 51***/
			mask = match_check(s_char,47,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,51,mask,0,0,1);
		}else if (src_state == 48){
			/***dest 49***/
			character_transitions_update(cfBV, ctof,48,49,mask,1,0,0);
			/***dest 52***/
			mask = match_check(s_char,48,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,52,mask,0,0,1);
			/***dest 53***/
			character_transitions_update(cfBV, ctof,48,53,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*53) + threadIdx.x] |= cfBV[ctof + (blockDim.x*48) + threadIdx.x];
			cfBV[ctof + (blockDim.x*58) + threadIdx.x] |= cfBV[ctof + (blockDim.x*48) + threadIdx.x];
			cfBV[ctof + (blockDim.x*63) + threadIdx.x] |= cfBV[ctof + (blockDim.x*48) + threadIdx.x];
		}else if (src_state == 49){
			/***dest 50***/
			character_transitions_update(cfBV, ctof,49,50,mask,1,0,0);
			/***dest 53***/
			mask = match_check(s_char,49,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,49,53,mask,0,0,1);
			/***dest 54***/
			character_transitions_update(cfBV, ctof,49,54,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*54) + threadIdx.x] |= cfBV[ctof + (blockDim.x*49) + threadIdx.x];
			cfBV[ctof + (blockDim.x*59) + threadIdx.x] |= cfBV[ctof + (blockDim.x*49) + threadIdx.x];
		}else if (src_state == 50){
			/***dest 51***/
			character_transitions_update(cfBV, ctof,50,51,mask,1,0,0);
			/***dest 54***/
			mask = match_check(s_char,50,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,54,mask,0,0,1);
			/***dest 55***/
			character_transitions_update(cfBV, ctof,50,55,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*55) + threadIdx.x] |= cfBV[ctof + (blockDim.x*50) + threadIdx.x];
		}else if (src_state == 51){
			/***dest 55***/
			mask = match_check(s_char,51,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,55,mask,0,0,1);
		}else if (src_state == 52){
			/***dest 53***/
			character_transitions_update(cfBV, ctof,52,53,mask,1,0,0);
			/***dest 56***/
			mask = match_check(s_char,52,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,56,mask,0,0,1);
			/***dest 57***/
			character_transitions_update(cfBV, ctof,52,57,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*57) + threadIdx.x] |= cfBV[ctof + (blockDim.x*52) + threadIdx.x];
			cfBV[ctof + (blockDim.x*62) + threadIdx.x] |= cfBV[ctof + (blockDim.x*52) + threadIdx.x];
			cfBV[ctof + (blockDim.x*67) + threadIdx.x] |= cfBV[ctof + (blockDim.x*52) + threadIdx.x];
		}else if (src_state == 53){
			/***dest 54***/
			character_transitions_update(cfBV, ctof,53,54,mask,1,0,0);
			/***dest 57***/
			mask = match_check(s_char,53,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,57,mask,0,0,1);
			/***dest 58***/
			character_transitions_update(cfBV, ctof,53,58,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*58) + threadIdx.x] |= cfBV[ctof + (blockDim.x*53) + threadIdx.x];
			cfBV[ctof + (blockDim.x*63) + threadIdx.x] |= cfBV[ctof + (blockDim.x*53) + threadIdx.x];
		}else if (src_state == 54){
			/***dest 55***/
			character_transitions_update(cfBV, ctof,54,55,mask,1,0,0);
			/***dest 58***/
			mask = match_check(s_char,54,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,58,mask,0,0,1);
			/***dest 59***/
			character_transitions_update(cfBV, ctof,54,59,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*59) + threadIdx.x] |= cfBV[ctof + (blockDim.x*54) + threadIdx.x];
		}else if (src_state == 55){
			/***dest 59***/
			mask = match_check(s_char,55,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,59,mask,0,0,1);
		}else if (src_state == 56){
			/***dest 57***/
			character_transitions_update(cfBV, ctof,56,57,mask,1,0,0);
			/***dest 60***/
			mask = match_check(s_char,56,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,60,mask,0,0,1);
			/***dest 61***/
			character_transitions_update(cfBV, ctof,56,61,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*61) + threadIdx.x] |= cfBV[ctof + (blockDim.x*56) + threadIdx.x];
			cfBV[ctof + (blockDim.x*66) + threadIdx.x] |= cfBV[ctof + (blockDim.x*56) + threadIdx.x];
			cfBV[ctof + (blockDim.x*71) + threadIdx.x] |= cfBV[ctof + (blockDim.x*56) + threadIdx.x];
		}else if (src_state == 57){
			/***dest 58***/
			character_transitions_update(cfBV, ctof,57,58,mask,1,0,0);
			/***dest 61***/
			mask = match_check(s_char,57,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,61,mask,0,0,1);
			/***dest 62***/
			character_transitions_update(cfBV, ctof,57,62,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*62) + threadIdx.x] |= cfBV[ctof + (blockDim.x*57) + threadIdx.x];
			cfBV[ctof + (blockDim.x*67) + threadIdx.x] |= cfBV[ctof + (blockDim.x*57) + threadIdx.x];
		}else if (src_state == 58){
			/***dest 59***/
			character_transitions_update(cfBV, ctof,58,59,mask,1,0,0);
			/***dest 62***/
			mask = match_check(s_char,58,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,62,mask,0,0,1);
			/***dest 63***/
			character_transitions_update(cfBV, ctof,58,63,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*63) + threadIdx.x] |= cfBV[ctof + (blockDim.x*58) + threadIdx.x];
		}else if (src_state == 59){
			/***dest 63***/
			mask = match_check(s_char,59,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,59,63,mask,0,0,1);
		}else if (src_state == 60){
			/***dest 61***/
			character_transitions_update(cfBV, ctof,60,61,mask,1,0,0);
			/***dest 64***/
			mask = match_check(s_char,60,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,60,64,mask,0,0,1);
			/***dest 65***/
			character_transitions_update(cfBV, ctof,60,65,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*65) + threadIdx.x] |= cfBV[ctof + (blockDim.x*60) + threadIdx.x];
			cfBV[ctof + (blockDim.x*70) + threadIdx.x] |= cfBV[ctof + (blockDim.x*60) + threadIdx.x];
			cfBV[ctof + (blockDim.x*75) + threadIdx.x] |= cfBV[ctof + (blockDim.x*60) + threadIdx.x];
		}else if (src_state == 61){
			/***dest 62***/
			character_transitions_update(cfBV, ctof,61,62,mask,1,0,0);
			/***dest 65***/
			mask = match_check(s_char,61,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,61,65,mask,0,0,1);
			/***dest 66***/
			character_transitions_update(cfBV, ctof,61,66,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*66) + threadIdx.x] |= cfBV[ctof + (blockDim.x*61) + threadIdx.x];
			cfBV[ctof + (blockDim.x*71) + threadIdx.x] |= cfBV[ctof + (blockDim.x*61) + threadIdx.x];
		}else if (src_state == 62){
			/***dest 63***/
			character_transitions_update(cfBV, ctof,62,63,mask,1,0,0);
			/***dest 66***/
			mask = match_check(s_char,62,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,66,mask,0,0,1);
			/***dest 67***/
			character_transitions_update(cfBV, ctof,62,67,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*67) + threadIdx.x] |= cfBV[ctof + (blockDim.x*62) + threadIdx.x];
		}else if (src_state == 63){
			/***dest 67***/
			mask = match_check(s_char,63,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,67,mask,0,0,1);
		}else if (src_state == 64){
			/***dest 65***/
			character_transitions_update(cfBV, ctof,64,65,mask,1,0,0);
			/***dest 68***/
			mask = match_check(s_char,64,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,68,mask,0,0,1);
			/***dest 69***/
			character_transitions_update(cfBV, ctof,64,69,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*69) + threadIdx.x] |= cfBV[ctof + (blockDim.x*64) + threadIdx.x];
			cfBV[ctof + (blockDim.x*74) + threadIdx.x] |= cfBV[ctof + (blockDim.x*64) + threadIdx.x];
			cfBV[ctof + (blockDim.x*79) + threadIdx.x] |= cfBV[ctof + (blockDim.x*64) + threadIdx.x];
		}else if (src_state == 65){
			/***dest 66***/
			character_transitions_update(cfBV, ctof,65,66,mask,1,0,0);
			/***dest 69***/
			mask = match_check(s_char,65,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,65,69,mask,0,0,1);
			/***dest 70***/
			character_transitions_update(cfBV, ctof,65,70,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*70) + threadIdx.x] |= cfBV[ctof + (blockDim.x*65) + threadIdx.x];
			cfBV[ctof + (blockDim.x*75) + threadIdx.x] |= cfBV[ctof + (blockDim.x*65) + threadIdx.x];
		}else if (src_state == 66){
			/***dest 67***/
			character_transitions_update(cfBV, ctof,66,67,mask,1,0,0);
			/***dest 70***/
			mask = match_check(s_char,66,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,66,70,mask,0,0,1);
			/***dest 71***/
			character_transitions_update(cfBV, ctof,66,71,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*71) + threadIdx.x] |= cfBV[ctof + (blockDim.x*66) + threadIdx.x];
		}else if (src_state == 67){
			/***dest 71***/
			mask = match_check(s_char,67,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,71,mask,0,0,1);
		}else if (src_state == 68){
			/***dest 69***/
			character_transitions_update(cfBV, ctof,68,69,mask,1,0,0);
			/***dest 72***/
			mask = match_check(s_char,68,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,68,72,mask,0,0,1);
			/***dest 73***/
			character_transitions_update(cfBV, ctof,68,73,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*73) + threadIdx.x] |= cfBV[ctof + (blockDim.x*68) + threadIdx.x];
			cfBV[ctof + (blockDim.x*78) + threadIdx.x] |= cfBV[ctof + (blockDim.x*68) + threadIdx.x];
			cfBV[ctof + (blockDim.x*83) + threadIdx.x] |= cfBV[ctof + (blockDim.x*68) + threadIdx.x];
		}else if (src_state == 69){
			/***dest 70***/
			character_transitions_update(cfBV, ctof,69,70,mask,1,0,0);
			/***dest 73***/
			mask = match_check(s_char,69,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,69,73,mask,0,0,1);
			/***dest 74***/
			character_transitions_update(cfBV, ctof,69,74,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*74) + threadIdx.x] |= cfBV[ctof + (blockDim.x*69) + threadIdx.x];
			cfBV[ctof + (blockDim.x*79) + threadIdx.x] |= cfBV[ctof + (blockDim.x*69) + threadIdx.x];
		}else if (src_state == 70){
			/***dest 71***/
			character_transitions_update(cfBV, ctof,70,71,mask,1,0,0);
			/***dest 74***/
			mask = match_check(s_char,70,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,74,mask,0,0,1);
			/***dest 75***/
			character_transitions_update(cfBV, ctof,70,75,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*75) + threadIdx.x] |= cfBV[ctof + (blockDim.x*70) + threadIdx.x];
		}else if (src_state == 71){
			/***dest 75***/
			mask = match_check(s_char,71,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,71,75,mask,0,0,1);
		}else if (src_state == 72){
			/***dest 73***/
			character_transitions_update(cfBV, ctof,72,73,mask,1,0,0);
			/***dest 76***/
			mask = match_check(s_char,72,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,72,76,mask,0,0,1);
			/***dest 77***/
			character_transitions_update(cfBV, ctof,72,77,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*77) + threadIdx.x] |= cfBV[ctof + (blockDim.x*72) + threadIdx.x];
			cfBV[ctof + (blockDim.x*82) + threadIdx.x] |= cfBV[ctof + (blockDim.x*72) + threadIdx.x];
		}else if (src_state == 73){
			/***dest 74***/
			character_transitions_update(cfBV, ctof,73,74,mask,1,0,0);
			/***dest 77***/
			mask = match_check(s_char,73,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,73,77,mask,0,0,1);
			/***dest 78***/
			character_transitions_update(cfBV, ctof,73,78,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*78) + threadIdx.x] |= cfBV[ctof + (blockDim.x*73) + threadIdx.x];
			cfBV[ctof + (blockDim.x*83) + threadIdx.x] |= cfBV[ctof + (blockDim.x*73) + threadIdx.x];
		}else if (src_state == 74){
			/***dest 75***/
			character_transitions_update(cfBV, ctof,74,75,mask,1,0,0);
			/***dest 78***/
			mask = match_check(s_char,74,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,74,78,mask,0,0,1);
			/***dest 79***/
			character_transitions_update(cfBV, ctof,74,79,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*79) + threadIdx.x] |= cfBV[ctof + (blockDim.x*74) + threadIdx.x];
		}else if (src_state == 75){
			/***dest 79***/
			mask = match_check(s_char,75,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,75,79,mask,0,0,1);
		}else if (src_state == 76){
			/***dest 77***/
			character_transitions_update(cfBV, ctof,76,77,mask,1,0,0);
			/***dest 80***/
			mask = match_check(s_char,76,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,76,80,mask,0,0,1);
			/***dest 81***/
			character_transitions_update(cfBV, ctof,76,81,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*81) + threadIdx.x] |= cfBV[ctof + (blockDim.x*76) + threadIdx.x];
		}else if (src_state == 77){
			/***dest 78***/
			character_transitions_update(cfBV, ctof,77,78,mask,1,0,0);
			/***dest 81***/
			mask = match_check(s_char,77,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,77,81,mask,0,0,1);
			/***dest 82***/
			character_transitions_update(cfBV, ctof,77,82,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*82) + threadIdx.x] |= cfBV[ctof + (blockDim.x*77) + threadIdx.x];
		}else if (src_state == 78){
			/***dest 79***/
			character_transitions_update(cfBV, ctof,78,79,mask,1,0,0);
			/***dest 82***/
			mask = match_check(s_char,78,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,78,82,mask,0,0,1);
			/***dest 83***/
			character_transitions_update(cfBV, ctof,78,83,mask,1,0,0);
			/***epsilon tx handling***/
			cfBV[ctof + (blockDim.x*83) + threadIdx.x] |= cfBV[ctof + (blockDim.x*78) + threadIdx.x];
		}else if (src_state == 79){
			/***dest 83***/
			mask = match_check(s_char,79,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,79,83,mask,0,0,1);
		}else if (src_state == 80){
			/***dest 81***/
			character_transitions_update(cfBV, ctof,80,81,mask,1,0,0);
		}else if (src_state == 81){
			/***dest 82***/
			character_transitions_update(cfBV, ctof,81,82,mask,1,0,0);
		}else if (src_state == 82){
			/***dest 83***/
			character_transitions_update(cfBV, ctof,82,83,mask,1,0,0);
		}else if (src_state == 83){
		}
	}
#endif

#ifdef Hamming_k8_d1_kernel
	for(src_state = 0; src_state < 17; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,1,0);
		}else if (src_state == 1){
			/***dest 3***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,3,mask,0,0,1);
			/***dest 4***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,4,mask,0,1,0);
		}else if (src_state == 2){
			/***dest 4***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,4,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 5***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,5,mask,0,0,1);
			/***dest 6***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,6,mask,0,1,0);
		}else if (src_state == 4){
			/***dest 6***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,6,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 7***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,7,mask,0,0,1);
			/***dest 8***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,8,mask,0,1,0);
		}else if (src_state == 6){
			/***dest 8***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,8,mask,0,0,1);
		}else if (src_state == 7){
			/***dest 9***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,9,mask,0,0,1);
			/***dest 10***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,10,mask,0,1,0);
		}else if (src_state == 8){
			/***dest 10***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,10,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 11***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,11,mask,0,0,1);
			/***dest 12***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,12,mask,0,1,0);
		}else if (src_state == 10){
			/***dest 12***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,12,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 13***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,13,mask,0,0,1);
			/***dest 14***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,14,mask,0,1,0);
		}else if (src_state == 12){
			/***dest 14***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,14,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 15***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,15,mask,0,0,1);
			/***dest 16***/
			mask = match_check(s_char,21,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,16,mask,0,1,0);
		}else if (src_state == 14){
			/***dest 16***/
			mask = match_check(s_char,22,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,16,mask,0,0,1);
		}else if (src_state == 15){
		}else if (src_state == 16){
		}
	}
#endif

#ifdef Hamming_k20_d3_kernel
	for(src_state = 0; src_state < 78; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,1,0);
		}else if (src_state == 1){
			/***dest 3***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,3,mask,0,0,1);
			/***dest 4***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,4,mask,0,1,0);
		}else if (src_state == 2){
			/***dest 4***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,4,mask,0,0,1);
			/***dest 5***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,5,mask,0,1,0);
		}else if (src_state == 3){
			/***dest 6***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,6,mask,0,0,1);
			/***dest 7***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,7,mask,0,1,0);
		}else if (src_state == 4){
			/***dest 7***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,7,mask,0,0,1);
			/***dest 8***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,8,mask,0,1,0);
		}else if (src_state == 5){
			/***dest 8***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,8,mask,0,0,1);
			/***dest 9***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,9,mask,0,1,0);
		}else if (src_state == 6){
			/***dest 10***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,10,mask,0,0,1);
			/***dest 11***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,11,mask,0,1,0);
		}else if (src_state == 7){
			/***dest 11***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,11,mask,0,0,1);
			/***dest 12***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,12,mask,0,1,0);
		}else if (src_state == 8){
			/***dest 12***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,12,mask,0,0,1);
			/***dest 13***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,13,mask,0,1,0);
		}else if (src_state == 9){
			/***dest 13***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,13,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 14***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,14,mask,0,0,1);
			/***dest 15***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,15,mask,0,1,0);
		}else if (src_state == 11){
			/***dest 15***/
			mask = match_check(s_char,21,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,15,mask,0,0,1);
			/***dest 16***/
			mask = match_check(s_char,22,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,16,mask,0,1,0);
		}else if (src_state == 12){
			/***dest 16***/
			mask = match_check(s_char,23,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,16,mask,0,0,1);
			/***dest 17***/
			mask = match_check(s_char,24,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,17,mask,0,1,0);
		}else if (src_state == 13){
			/***dest 17***/
			mask = match_check(s_char,25,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,17,mask,0,0,1);
		}else if (src_state == 14){
			/***dest 18***/
			mask = match_check(s_char,26,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,18,mask,0,0,1);
			/***dest 19***/
			mask = match_check(s_char,27,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,19,mask,0,1,0);
		}else if (src_state == 15){
			/***dest 19***/
			mask = match_check(s_char,28,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,19,mask,0,0,1);
			/***dest 20***/
			mask = match_check(s_char,29,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,20,mask,0,1,0);
		}else if (src_state == 16){
			/***dest 20***/
			mask = match_check(s_char,30,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,20,mask,0,0,1);
			/***dest 21***/
			mask = match_check(s_char,31,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,21,mask,0,1,0);
		}else if (src_state == 17){
			/***dest 21***/
			mask = match_check(s_char,32,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,21,mask,0,0,1);
		}else if (src_state == 18){
			/***dest 22***/
			mask = match_check(s_char,33,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,22,mask,0,0,1);
			/***dest 23***/
			mask = match_check(s_char,34,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,23,mask,0,1,0);
		}else if (src_state == 19){
			/***dest 23***/
			mask = match_check(s_char,35,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,23,mask,0,0,1);
			/***dest 24***/
			mask = match_check(s_char,36,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,24,mask,0,1,0);
		}else if (src_state == 20){
			/***dest 24***/
			mask = match_check(s_char,37,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,24,mask,0,0,1);
			/***dest 25***/
			mask = match_check(s_char,38,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,25,mask,0,1,0);
		}else if (src_state == 21){
			/***dest 25***/
			mask = match_check(s_char,39,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,21,25,mask,0,0,1);
		}else if (src_state == 22){
			/***dest 26***/
			mask = match_check(s_char,40,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,26,mask,0,0,1);
			/***dest 27***/
			mask = match_check(s_char,41,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,27,mask,0,1,0);
		}else if (src_state == 23){
			/***dest 27***/
			mask = match_check(s_char,42,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,23,27,mask,0,0,1);
			/***dest 28***/
			mask = match_check(s_char,43,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,23,28,mask,0,1,0);
		}else if (src_state == 24){
			/***dest 28***/
			mask = match_check(s_char,44,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,28,mask,0,0,1);
			/***dest 29***/
			mask = match_check(s_char,45,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,29,mask,0,1,0);
		}else if (src_state == 25){
			/***dest 29***/
			mask = match_check(s_char,46,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,25,29,mask,0,0,1);
		}else if (src_state == 26){
			/***dest 30***/
			mask = match_check(s_char,47,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,30,mask,0,0,1);
			/***dest 31***/
			mask = match_check(s_char,48,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,31,mask,0,1,0);
		}else if (src_state == 27){
			/***dest 31***/
			mask = match_check(s_char,49,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,27,31,mask,0,0,1);
			/***dest 32***/
			mask = match_check(s_char,50,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,27,32,mask,0,1,0);
		}else if (src_state == 28){
			/***dest 32***/
			mask = match_check(s_char,51,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,32,mask,0,0,1);
			/***dest 33***/
			mask = match_check(s_char,52,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,33,mask,0,1,0);
		}else if (src_state == 29){
			/***dest 33***/
			mask = match_check(s_char,53,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,33,mask,0,0,1);
		}else if (src_state == 30){
			/***dest 34***/
			mask = match_check(s_char,54,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,34,mask,0,0,1);
			/***dest 35***/
			mask = match_check(s_char,55,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,35,mask,0,1,0);
		}else if (src_state == 31){
			/***dest 35***/
			mask = match_check(s_char,56,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,35,mask,0,0,1);
			/***dest 36***/
			mask = match_check(s_char,57,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,36,mask,0,1,0);
		}else if (src_state == 32){
			/***dest 36***/
			mask = match_check(s_char,58,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,32,36,mask,0,0,1);
			/***dest 37***/
			mask = match_check(s_char,59,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,32,37,mask,0,1,0);
		}else if (src_state == 33){
			/***dest 37***/
			mask = match_check(s_char,60,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,33,37,mask,0,0,1);
		}else if (src_state == 34){
			/***dest 38***/
			mask = match_check(s_char,61,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,34,38,mask,0,0,1);
			/***dest 39***/
			mask = match_check(s_char,62,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,34,39,mask,0,1,0);
		}else if (src_state == 35){
			/***dest 39***/
			mask = match_check(s_char,63,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,39,mask,0,0,1);
			/***dest 40***/
			mask = match_check(s_char,64,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,40,mask,0,1,0);
		}else if (src_state == 36){
			/***dest 40***/
			mask = match_check(s_char,65,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,36,40,mask,0,0,1);
			/***dest 41***/
			mask = match_check(s_char,66,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,36,41,mask,0,1,0);
		}else if (src_state == 37){
			/***dest 41***/
			mask = match_check(s_char,67,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,41,mask,0,0,1);
		}else if (src_state == 38){
			/***dest 42***/
			mask = match_check(s_char,68,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,38,42,mask,0,0,1);
			/***dest 43***/
			mask = match_check(s_char,69,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,38,43,mask,0,1,0);
		}else if (src_state == 39){
			/***dest 43***/
			mask = match_check(s_char,70,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,43,mask,0,0,1);
			/***dest 44***/
			mask = match_check(s_char,71,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,44,mask,0,1,0);
		}else if (src_state == 40){
			/***dest 44***/
			mask = match_check(s_char,72,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,44,mask,0,0,1);
			/***dest 45***/
			mask = match_check(s_char,73,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,45,mask,0,1,0);
		}else if (src_state == 41){
			/***dest 45***/
			mask = match_check(s_char,74,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,45,mask,0,0,1);
		}else if (src_state == 42){
			/***dest 46***/
			mask = match_check(s_char,75,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,46,mask,0,0,1);
			/***dest 47***/
			mask = match_check(s_char,76,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,47,mask,0,1,0);
		}else if (src_state == 43){
			/***dest 47***/
			mask = match_check(s_char,77,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,47,mask,0,0,1);
			/***dest 48***/
			mask = match_check(s_char,78,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,48,mask,0,1,0);
		}else if (src_state == 44){
			/***dest 48***/
			mask = match_check(s_char,79,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,48,mask,0,0,1);
			/***dest 49***/
			mask = match_check(s_char,80,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,49,mask,0,1,0);
		}else if (src_state == 45){
			/***dest 49***/
			mask = match_check(s_char,81,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,49,mask,0,0,1);
		}else if (src_state == 46){
			/***dest 50***/
			mask = match_check(s_char,82,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,50,mask,0,0,1);
			/***dest 51***/
			mask = match_check(s_char,83,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,51,mask,0,1,0);
		}else if (src_state == 47){
			/***dest 51***/
			mask = match_check(s_char,84,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,51,mask,0,0,1);
			/***dest 52***/
			mask = match_check(s_char,85,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,52,mask,0,1,0);
		}else if (src_state == 48){
			/***dest 52***/
			mask = match_check(s_char,86,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,52,mask,0,0,1);
			/***dest 53***/
			mask = match_check(s_char,87,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,53,mask,0,1,0);
		}else if (src_state == 49){
			/***dest 53***/
			mask = match_check(s_char,88,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,49,53,mask,0,0,1);
		}else if (src_state == 50){
			/***dest 54***/
			mask = match_check(s_char,89,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,54,mask,0,0,1);
			/***dest 55***/
			mask = match_check(s_char,90,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,55,mask,0,1,0);
		}else if (src_state == 51){
			/***dest 55***/
			mask = match_check(s_char,91,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,55,mask,0,0,1);
			/***dest 56***/
			mask = match_check(s_char,92,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,56,mask,0,1,0);
		}else if (src_state == 52){
			/***dest 56***/
			mask = match_check(s_char,93,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,56,mask,0,0,1);
			/***dest 57***/
			mask = match_check(s_char,94,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,57,mask,0,1,0);
		}else if (src_state == 53){
			/***dest 57***/
			mask = match_check(s_char,95,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,57,mask,0,0,1);
		}else if (src_state == 54){
			/***dest 58***/
			mask = match_check(s_char,96,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,58,mask,0,0,1);
			/***dest 59***/
			mask = match_check(s_char,97,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,59,mask,0,1,0);
		}else if (src_state == 55){
			/***dest 59***/
			mask = match_check(s_char,98,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,59,mask,0,0,1);
			/***dest 60***/
			mask = match_check(s_char,99,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,60,mask,0,1,0);
		}else if (src_state == 56){
			/***dest 60***/
			mask = match_check(s_char,100,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,60,mask,0,0,1);
			/***dest 61***/
			mask = match_check(s_char,101,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,61,mask,0,1,0);
		}else if (src_state == 57){
			/***dest 61***/
			mask = match_check(s_char,102,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,61,mask,0,0,1);
		}else if (src_state == 58){
			/***dest 62***/
			mask = match_check(s_char,103,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,62,mask,0,0,1);
			/***dest 63***/
			mask = match_check(s_char,104,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,63,mask,0,1,0);
		}else if (src_state == 59){
			/***dest 63***/
			mask = match_check(s_char,105,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,59,63,mask,0,0,1);
			/***dest 64***/
			mask = match_check(s_char,106,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,59,64,mask,0,1,0);
		}else if (src_state == 60){
			/***dest 64***/
			mask = match_check(s_char,107,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,60,64,mask,0,0,1);
			/***dest 65***/
			mask = match_check(s_char,108,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,60,65,mask,0,1,0);
		}else if (src_state == 61){
			/***dest 65***/
			mask = match_check(s_char,109,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,61,65,mask,0,0,1);
		}else if (src_state == 62){
			/***dest 66***/
			mask = match_check(s_char,110,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,66,mask,0,0,1);
			/***dest 67***/
			mask = match_check(s_char,111,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,67,mask,0,1,0);
		}else if (src_state == 63){
			/***dest 67***/
			mask = match_check(s_char,112,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,67,mask,0,0,1);
			/***dest 68***/
			mask = match_check(s_char,113,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,68,mask,0,1,0);
		}else if (src_state == 64){
			/***dest 68***/
			mask = match_check(s_char,114,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,68,mask,0,0,1);
			/***dest 69***/
			mask = match_check(s_char,115,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,69,mask,0,1,0);
		}else if (src_state == 65){
			/***dest 69***/
			mask = match_check(s_char,116,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,65,69,mask,0,0,1);
		}else if (src_state == 66){
			/***dest 70***/
			mask = match_check(s_char,117,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,66,70,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,118,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,66,71,mask,0,1,0);
		}else if (src_state == 67){
			/***dest 71***/
			mask = match_check(s_char,119,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,71,mask,0,0,1);
			/***dest 72***/
			mask = match_check(s_char,120,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,72,mask,0,1,0);
		}else if (src_state == 68){
			/***dest 72***/
			mask = match_check(s_char,121,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,68,72,mask,0,0,1);
			/***dest 73***/
			mask = match_check(s_char,122,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,68,73,mask,0,1,0);
		}else if (src_state == 69){
			/***dest 73***/
			mask = match_check(s_char,123,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,69,73,mask,0,0,1);
		}else if (src_state == 70){
			/***dest 74***/
			mask = match_check(s_char,124,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,74,mask,0,0,1);
			/***dest 75***/
			mask = match_check(s_char,125,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,75,mask,0,1,0);
		}else if (src_state == 71){
			/***dest 75***/
			mask = match_check(s_char,126,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,71,75,mask,0,0,1);
			/***dest 76***/
			mask = match_check(s_char,127,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,71,76,mask,0,1,0);
		}else if (src_state == 72){
			/***dest 76***/
			mask = match_check(s_char,128,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,72,76,mask,0,0,1);
			/***dest 77***/
			mask = match_check(s_char,129,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,72,77,mask,0,1,0);
		}else if (src_state == 73){
			/***dest 77***/
			mask = match_check(s_char,130,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,73,77,mask,0,0,1);
		}else if (src_state == 74){
		}else if (src_state == 75){
		}else if (src_state == 76){
		}else if (src_state == 77){
		}
	}
#endif

#ifdef Fermi_kernel
	for(src_state = 0; src_state < 18; src_state++){
 if (src_state == 0){
			/***dest 1***/
			character_transitions_update(cfBV, ctof,0,1,mask,1,0,0);
		}else if (src_state == 1){
			/***dest 4***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,4,mask,0,0,1);
			/***dest 2***/
			character_transitions_update(cfBV, ctof,1,2,mask,1,0,0);
		}else if (src_state == 2){
			/***dest 3***/
			character_transitions_update(cfBV, ctof,2,3,mask,1,0,0);
		}else if (src_state == 3){
			/***dest 4***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,4,mask,0,0,1);
			/***dest 2***/
			character_transitions_update(cfBV, ctof,3,2,mask,1,0,0);
		}else if (src_state == 4){
			/***dest 5***/
			mask = match_check(s_char,2,4,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,5,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 8***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,8,mask,0,0,1);
			/***dest 6***/
			character_transitions_update(cfBV, ctof,5,6,mask,1,0,0);
		}else if (src_state == 6){
			/***dest 7***/
			character_transitions_update(cfBV, ctof,6,7,mask,1,0,0);
		}else if (src_state == 7){
			/***dest 8***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,8,mask,0,0,1);
			/***dest 6***/
			character_transitions_update(cfBV, ctof,7,6,mask,1,0,0);
		}else if (src_state == 8){
			/***dest 9***/
			mask = match_check(s_char,8,4,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,9,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 10***/
			character_transitions_update(cfBV, ctof,9,10,mask,1,0,0);
			/***dest 12***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,12,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 11***/
			character_transitions_update(cfBV, ctof,10,11,mask,1,0,0);
		}else if (src_state == 11){
			/***dest 10***/
			character_transitions_update(cfBV, ctof,11,10,mask,1,0,0);
			/***dest 12***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,12,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 13***/
			mask = match_check(s_char,14,4,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,13,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 16***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,16,mask,0,0,1);
			/***dest 14***/
			character_transitions_update(cfBV, ctof,13,14,mask,1,0,0);
		}else if (src_state == 14){
			/***dest 15***/
			character_transitions_update(cfBV, ctof,14,15,mask,1,0,0);
		}else if (src_state == 15){
			/***dest 16***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,16,mask,0,0,1);
			/***dest 14***/
			character_transitions_update(cfBV, ctof,15,14,mask,1,0,0);
		}else if (src_state == 16){
			/***dest 17***/
			mask = match_check(s_char,20,4,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,17,mask,0,0,1);
		}else if (src_state == 17){
		}
	}
#endif

#ifdef SPM_kernel
	for(src_state = 0; src_state < 21; src_state++){
 if (src_state == 0){
			/***dest 5***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,5,mask,0,0,1);
			/***dest 1***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 8***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,8,mask,0,0,1);
			/***dest 11***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,11,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 2***/
			mask = match_check(s_char,4,2,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,2,mask,0,1,0);
			/***dest 3***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,3,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 1***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,8,2,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,2,mask,0,1,0);
			/***dest 3***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,3,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 4***/
			mask = match_check(s_char,11,3,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,4,mask,0,1,0);
			/***dest 6***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,6,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 5***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,5,mask,0,0,1);
			/***dest 4***/
			mask = match_check(s_char,16,3,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,4,mask,0,1,0);
			/***dest 6***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,6,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 4***/
			mask = match_check(s_char,20,3,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,4,mask,0,1,0);
			/***dest 6***/
			mask = match_check(s_char,23,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,6,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 7***/
			mask = match_check(s_char,24,2,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,7,mask,0,1,0);
			/***dest 9***/
			mask = match_check(s_char,26,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,9,mask,0,0,1);
		}else if (src_state == 7){
			/***dest 7***/
			mask = match_check(s_char,27,2,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,7,mask,0,1,0);
			/***dest 8***/
			mask = match_check(s_char,29,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,8,mask,0,0,1);
			/***dest 9***/
			mask = match_check(s_char,30,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,9,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 7***/
			mask = match_check(s_char,31,2,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,7,mask,0,1,0);
			/***dest 9***/
			mask = match_check(s_char,33,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,9,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 10***/
			mask = match_check(s_char,34,3,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,10,mask,0,1,0);
			/***dest 12***/
			mask = match_check(s_char,37,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,12,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 10***/
			mask = match_check(s_char,38,3,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,10,mask,0,1,0);
			/***dest 11***/
			mask = match_check(s_char,41,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,11,mask,0,0,1);
			/***dest 12***/
			mask = match_check(s_char,42,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,12,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 10***/
			mask = match_check(s_char,43,3,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,10,mask,0,1,0);
			/***dest 12***/
			mask = match_check(s_char,46,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,12,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 13***/
			mask = match_check(s_char,47,2,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,13,mask,0,1,0);
			/***dest 14***/
			mask = match_check(s_char,49,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,14,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 13***/
			mask = match_check(s_char,50,2,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,13,mask,0,1,0);
			/***dest 14***/
			mask = match_check(s_char,52,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,14,mask,0,0,1);
		}else if (src_state == 14){
			/***dest 15***/
			mask = match_check(s_char,53,3,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,15,mask,0,1,0);
			/***dest 16***/
			mask = match_check(s_char,56,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,16,mask,0,0,1);
		}else if (src_state == 15){
			/***dest 15***/
			mask = match_check(s_char,57,3,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,15,mask,0,1,0);
			/***dest 16***/
			mask = match_check(s_char,60,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,16,mask,0,0,1);
		}else if (src_state == 16){
			/***dest 17***/
			mask = match_check(s_char,61,2,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,17,mask,0,1,0);
			/***dest 18***/
			mask = match_check(s_char,63,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,18,mask,0,0,1);
		}else if (src_state == 17){
			/***dest 17***/
			mask = match_check(s_char,64,2,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,17,mask,0,1,0);
			/***dest 18***/
			mask = match_check(s_char,66,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,18,mask,0,0,1);
		}else if (src_state == 18){
			/***dest 19***/
			character_transitions_update(cfBV, ctof,18,19,mask,1,0,0);
			/***dest 20***/
			mask = match_check(s_char,67,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,20,mask,0,0,1);
		}else if (src_state == 19){
			/***dest 19***/
			character_transitions_update(cfBV, ctof,19,19,mask,1,0,0);
			/***dest 20***/
			mask = match_check(s_char,68,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,20,mask,0,0,1);
		}else if (src_state == 20){
		}
	}
#endif

#ifdef ER_kernel
	for(src_state = 0; src_state < 72; src_state++){
 if (src_state == 0){
			/***dest 0***/
			character_transitions_update(cfBV, ctof,0,0,mask,1,0,0);
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,1,0);
		}else if (src_state == 1){
			/***dest 3***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,3,mask,0,0,1);
			/***dest 4***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,4,mask,0,1,0);
		}else if (src_state == 2){
			/***dest 4***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,4,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 5***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,5,mask,0,0,1);
			/***dest 6***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,6,mask,0,1,0);
		}else if (src_state == 4){
			/***dest 6***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,6,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 7***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,7,mask,0,0,1);
			/***dest 8***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,8,mask,0,1,0);
			/***dest 23***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,34,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 8***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,8,mask,0,0,1);
			/***dest 23***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,34,mask,0,0,1);
		}else if (src_state == 7){
			/***dest 9***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,9,mask,0,0,1);
			/***dest 10***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,10,mask,0,1,0);
			/***dest 23***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,34,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 10***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,10,mask,0,0,1);
			/***dest 23***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,21,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,34,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 11***/
			mask = match_check(s_char,22,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,11,mask,0,0,1);
			/***dest 12***/
			mask = match_check(s_char,23,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,12,mask,0,1,0);
			/***dest 23***/
			mask = match_check(s_char,24,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,25,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,34,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 12***/
			mask = match_check(s_char,26,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,12,mask,0,0,1);
			/***dest 23***/
			mask = match_check(s_char,27,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,28,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,34,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 13***/
			mask = match_check(s_char,29,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,13,mask,0,0,1);
			/***dest 14***/
			mask = match_check(s_char,30,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,14,mask,0,1,0);
			/***dest 23***/
			mask = match_check(s_char,31,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,32,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,34,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 14***/
			mask = match_check(s_char,33,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,14,mask,0,0,1);
			/***dest 23***/
			mask = match_check(s_char,34,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,35,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,34,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 15***/
			mask = match_check(s_char,36,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,15,mask,0,0,1);
			/***dest 16***/
			mask = match_check(s_char,37,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,16,mask,0,1,0);
			/***dest 23***/
			mask = match_check(s_char,38,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,39,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,34,mask,0,0,1);
		}else if (src_state == 14){
			/***dest 16***/
			mask = match_check(s_char,40,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,16,mask,0,0,1);
			/***dest 23***/
			mask = match_check(s_char,41,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,42,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,34,mask,0,0,1);
		}else if (src_state == 15){
			/***dest 17***/
			mask = match_check(s_char,43,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,17,mask,0,0,1);
			/***dest 18***/
			mask = match_check(s_char,44,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,18,mask,0,1,0);
			/***dest 23***/
			mask = match_check(s_char,45,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,46,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,34,mask,0,0,1);
		}else if (src_state == 16){
			/***dest 18***/
			mask = match_check(s_char,47,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,18,mask,0,0,1);
			/***dest 23***/
			mask = match_check(s_char,48,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,49,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,34,mask,0,0,1);
		}else if (src_state == 17){
			/***dest 19***/
			mask = match_check(s_char,50,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,19,mask,0,0,1);
			/***dest 20***/
			mask = match_check(s_char,51,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,20,mask,0,1,0);
			/***dest 23***/
			mask = match_check(s_char,52,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,53,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,34,mask,0,0,1);
		}else if (src_state == 18){
			/***dest 20***/
			mask = match_check(s_char,54,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,20,mask,0,0,1);
			/***dest 23***/
			mask = match_check(s_char,55,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,56,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,34,mask,0,0,1);
		}else if (src_state == 19){
			/***dest 21***/
			mask = match_check(s_char,57,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,21,mask,0,0,1);
			/***dest 22***/
			mask = match_check(s_char,58,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,22,mask,0,1,0);
			/***dest 23***/
			mask = match_check(s_char,59,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,60,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,34,mask,0,0,1);
		}else if (src_state == 20){
			/***dest 22***/
			mask = match_check(s_char,61,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,22,mask,0,0,1);
			/***dest 23***/
			mask = match_check(s_char,62,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,63,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,34,mask,0,0,1);
		}else if (src_state == 21){
			/***dest 23***/
			mask = match_check(s_char,64,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,21,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,65,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,21,34,mask,0,0,1);
		}else if (src_state == 22){
			/***dest 23***/
			mask = match_check(s_char,66,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,23,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,67,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,34,mask,0,0,1);
		}else if (src_state == 23){
			/***dest 24***/
			mask = match_check(s_char,68,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,23,24,mask,0,0,1);
			/***dest 25***/
			mask = match_check(s_char,69,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,23,25,mask,0,1,0);
		}else if (src_state == 24){
			/***dest 26***/
			mask = match_check(s_char,70,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,26,mask,0,0,1);
			/***dest 27***/
			mask = match_check(s_char,71,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,27,mask,0,1,0);
		}else if (src_state == 25){
			/***dest 27***/
			mask = match_check(s_char,72,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,25,27,mask,0,0,1);
		}else if (src_state == 26){
			/***dest 28***/
			mask = match_check(s_char,73,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,28,mask,0,0,1);
			/***dest 29***/
			mask = match_check(s_char,74,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,29,mask,0,1,0);
		}else if (src_state == 27){
			/***dest 29***/
			mask = match_check(s_char,75,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,27,29,mask,0,0,1);
		}else if (src_state == 28){
			/***dest 30***/
			mask = match_check(s_char,76,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,30,mask,0,0,1);
			/***dest 31***/
			mask = match_check(s_char,77,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,31,mask,0,1,0);
			/***dest 34***/
			mask = match_check(s_char,78,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,34,mask,0,0,1);
			/***dest 68***/
			mask = match_check(s_char,79,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,68,mask,0,1,0);
		}else if (src_state == 29){
			/***dest 31***/
			mask = match_check(s_char,80,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,31,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,81,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,34,mask,0,0,1);
			/***dest 68***/
			mask = match_check(s_char,82,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,68,mask,0,1,0);
		}else if (src_state == 30){
			/***dest 32***/
			mask = match_check(s_char,83,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,32,mask,0,0,1);
			/***dest 33***/
			mask = match_check(s_char,84,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,33,mask,0,1,0);
			/***dest 34***/
			mask = match_check(s_char,85,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,34,mask,0,0,1);
			/***dest 68***/
			mask = match_check(s_char,86,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,68,mask,0,1,0);
		}else if (src_state == 31){
			/***dest 33***/
			mask = match_check(s_char,87,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,33,mask,0,0,1);
			/***dest 34***/
			mask = match_check(s_char,88,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,34,mask,0,0,1);
			/***dest 68***/
			mask = match_check(s_char,89,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,68,mask,0,1,0);
		}else if (src_state == 32){
			/***dest 34***/
			mask = match_check(s_char,90,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,32,34,mask,0,0,1);
			/***dest 68***/
			mask = match_check(s_char,91,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,32,68,mask,0,1,0);
		}else if (src_state == 33){
			/***dest 34***/
			mask = match_check(s_char,92,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,33,34,mask,0,0,1);
			/***dest 68***/
			mask = match_check(s_char,93,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,33,68,mask,0,1,0);
		}else if (src_state == 34){
			/***dest 35***/
			mask = match_check(s_char,94,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,34,35,mask,0,0,1);
			/***dest 36***/
			mask = match_check(s_char,95,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,34,36,mask,0,1,0);
		}else if (src_state == 35){
			/***dest 37***/
			mask = match_check(s_char,96,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,37,mask,0,0,1);
			/***dest 38***/
			mask = match_check(s_char,97,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,38,mask,0,1,0);
		}else if (src_state == 36){
			/***dest 38***/
			mask = match_check(s_char,98,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,36,38,mask,0,0,1);
		}else if (src_state == 37){
			/***dest 39***/
			mask = match_check(s_char,99,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,39,mask,0,0,1);
			/***dest 40***/
			mask = match_check(s_char,100,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,40,mask,0,1,0);
		}else if (src_state == 38){
			/***dest 40***/
			mask = match_check(s_char,101,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,38,40,mask,0,0,1);
		}else if (src_state == 39){
			/***dest 41***/
			mask = match_check(s_char,102,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,41,mask,0,0,1);
			/***dest 42***/
			mask = match_check(s_char,103,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,42,mask,0,1,0);
			/***dest 57***/
			mask = match_check(s_char,104,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,105,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,71,mask,0,0,1);
		}else if (src_state == 40){
			/***dest 42***/
			mask = match_check(s_char,106,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,42,mask,0,0,1);
			/***dest 57***/
			mask = match_check(s_char,107,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,108,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,71,mask,0,0,1);
		}else if (src_state == 41){
			/***dest 43***/
			mask = match_check(s_char,109,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,43,mask,0,0,1);
			/***dest 44***/
			mask = match_check(s_char,110,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,44,mask,0,1,0);
			/***dest 57***/
			mask = match_check(s_char,111,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,112,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,71,mask,0,0,1);
		}else if (src_state == 42){
			/***dest 44***/
			mask = match_check(s_char,113,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,44,mask,0,0,1);
			/***dest 57***/
			mask = match_check(s_char,114,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,115,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,71,mask,0,0,1);
		}else if (src_state == 43){
			/***dest 45***/
			mask = match_check(s_char,116,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,45,mask,0,0,1);
			/***dest 46***/
			mask = match_check(s_char,117,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,46,mask,0,1,0);
			/***dest 57***/
			mask = match_check(s_char,118,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,119,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,71,mask,0,0,1);
		}else if (src_state == 44){
			/***dest 46***/
			mask = match_check(s_char,120,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,46,mask,0,0,1);
			/***dest 57***/
			mask = match_check(s_char,121,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,122,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,71,mask,0,0,1);
		}else if (src_state == 45){
			/***dest 47***/
			mask = match_check(s_char,123,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,47,mask,0,0,1);
			/***dest 48***/
			mask = match_check(s_char,124,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,48,mask,0,1,0);
			/***dest 57***/
			mask = match_check(s_char,125,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,126,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,71,mask,0,0,1);
		}else if (src_state == 46){
			/***dest 48***/
			mask = match_check(s_char,127,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,48,mask,0,0,1);
			/***dest 57***/
			mask = match_check(s_char,128,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,129,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,71,mask,0,0,1);
		}else if (src_state == 47){
			/***dest 49***/
			mask = match_check(s_char,130,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,49,mask,0,0,1);
			/***dest 50***/
			mask = match_check(s_char,131,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,50,mask,0,1,0);
			/***dest 57***/
			mask = match_check(s_char,132,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,133,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,71,mask,0,0,1);
		}else if (src_state == 48){
			/***dest 50***/
			mask = match_check(s_char,134,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,50,mask,0,0,1);
			/***dest 57***/
			mask = match_check(s_char,135,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,136,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,71,mask,0,0,1);
		}else if (src_state == 49){
			/***dest 51***/
			mask = match_check(s_char,137,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,49,51,mask,0,0,1);
			/***dest 52***/
			mask = match_check(s_char,138,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,49,52,mask,0,1,0);
			/***dest 57***/
			mask = match_check(s_char,139,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,49,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,140,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,49,71,mask,0,0,1);
		}else if (src_state == 50){
			/***dest 52***/
			mask = match_check(s_char,141,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,52,mask,0,0,1);
			/***dest 57***/
			mask = match_check(s_char,142,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,143,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,71,mask,0,0,1);
		}else if (src_state == 51){
			/***dest 53***/
			mask = match_check(s_char,144,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,53,mask,0,0,1);
			/***dest 54***/
			mask = match_check(s_char,145,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,54,mask,0,1,0);
			/***dest 57***/
			mask = match_check(s_char,146,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,147,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,71,mask,0,0,1);
		}else if (src_state == 52){
			/***dest 54***/
			mask = match_check(s_char,148,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,54,mask,0,0,1);
			/***dest 57***/
			mask = match_check(s_char,149,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,150,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,71,mask,0,0,1);
		}else if (src_state == 53){
			/***dest 55***/
			mask = match_check(s_char,151,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,55,mask,0,0,1);
			/***dest 56***/
			mask = match_check(s_char,152,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,56,mask,0,1,0);
			/***dest 57***/
			mask = match_check(s_char,153,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,154,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,71,mask,0,0,1);
		}else if (src_state == 54){
			/***dest 56***/
			mask = match_check(s_char,155,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,56,mask,0,0,1);
			/***dest 57***/
			mask = match_check(s_char,156,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,157,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,71,mask,0,0,1);
		}else if (src_state == 55){
			/***dest 57***/
			mask = match_check(s_char,158,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,159,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,71,mask,0,0,1);
		}else if (src_state == 56){
			/***dest 57***/
			mask = match_check(s_char,160,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,57,mask,0,0,1);
			/***dest 71***/
			mask = match_check(s_char,161,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,71,mask,0,0,1);
		}else if (src_state == 57){
			/***dest 58***/
			mask = match_check(s_char,162,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,58,mask,0,0,1);
			/***dest 59***/
			mask = match_check(s_char,163,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,59,mask,0,1,0);
			/***dest 69***/
			mask = match_check(s_char,164,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,69,mask,0,0,1);
		}else if (src_state == 58){
			/***dest 60***/
			mask = match_check(s_char,165,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,60,mask,0,0,1);
			/***dest 61***/
			mask = match_check(s_char,166,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,61,mask,0,1,0);
		}else if (src_state == 59){
			/***dest 61***/
			mask = match_check(s_char,167,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,59,61,mask,0,0,1);
		}else if (src_state == 60){
			/***dest 62***/
			mask = match_check(s_char,168,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,60,62,mask,0,0,1);
			/***dest 63***/
			mask = match_check(s_char,169,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,60,63,mask,0,1,0);
		}else if (src_state == 61){
			/***dest 63***/
			mask = match_check(s_char,170,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,61,63,mask,0,0,1);
		}else if (src_state == 62){
			/***dest 64***/
			mask = match_check(s_char,171,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,64,mask,0,0,1);
			/***dest 65***/
			mask = match_check(s_char,172,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,65,mask,0,1,0);
			/***dest 70***/
			mask = match_check(s_char,173,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,70,mask,0,1,0);
			/***dest 71***/
			mask = match_check(s_char,174,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,71,mask,0,0,1);
		}else if (src_state == 63){
			/***dest 65***/
			mask = match_check(s_char,175,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,65,mask,0,0,1);
			/***dest 70***/
			mask = match_check(s_char,176,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,70,mask,0,1,0);
			/***dest 71***/
			mask = match_check(s_char,177,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,71,mask,0,0,1);
		}else if (src_state == 64){
			/***dest 66***/
			mask = match_check(s_char,178,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,66,mask,0,0,1);
			/***dest 67***/
			mask = match_check(s_char,179,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,67,mask,0,1,0);
			/***dest 70***/
			mask = match_check(s_char,180,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,70,mask,0,1,0);
			/***dest 71***/
			mask = match_check(s_char,181,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,71,mask,0,0,1);
		}else if (src_state == 65){
			/***dest 67***/
			mask = match_check(s_char,182,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,65,67,mask,0,0,1);
			/***dest 70***/
			mask = match_check(s_char,183,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,65,70,mask,0,1,0);
			/***dest 71***/
			mask = match_check(s_char,184,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,65,71,mask,0,0,1);
		}else if (src_state == 66){
			/***dest 70***/
			mask = match_check(s_char,185,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,66,70,mask,0,1,0);
			/***dest 71***/
			mask = match_check(s_char,186,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,66,71,mask,0,0,1);
		}else if (src_state == 67){
			/***dest 70***/
			mask = match_check(s_char,187,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,70,mask,0,1,0);
			/***dest 71***/
			mask = match_check(s_char,188,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,71,mask,0,0,1);
		}else if (src_state == 68){
			/***dest 34***/
			mask = match_check(s_char,189,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,68,34,mask,0,0,1);
			/***dest 68***/
			mask = match_check(s_char,190,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,68,68,mask,0,1,0);
		}else if (src_state == 69){
			/***dest 58***/
			mask = match_check(s_char,191,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,69,58,mask,0,0,1);
			/***dest 59***/
			mask = match_check(s_char,192,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,69,59,mask,0,1,0);
		}else if (src_state == 70){
			/***dest 70***/
			mask = match_check(s_char,193,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,70,mask,0,1,0);
			/***dest 71***/
			mask = match_check(s_char,194,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,71,mask,0,0,1);
		}else if (src_state == 71){
		}
	}
#endif

#ifdef	Synthetic_simple_20states_kernel
	for(src_state = 0; src_state < 20; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,0,1);
			/***dest 3***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,3,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 4***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,4,mask,0,0,1);
			/***dest 5***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,5,mask,0,0,1);
			/***dest 6***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 0***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,0,mask,0,0,1);
			/***dest 8***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 9***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 10***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 11***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,11,mask,0,0,1);
		}else if (src_state == 7){
			/***dest 12***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,12,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 13***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,13,mask,0,0,1);
			/***dest 14***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,14,mask,0,0,1);
			/***dest 15***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,15,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 16***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,16,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 17***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,17,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 18***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,18,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 19***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,19,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 8***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,8,mask,0,0,1);
		}else if (src_state == 14){
		}else if (src_state == 15){
		}else if (src_state == 16){
		}else if (src_state == 17){
		}else if (src_state == 18){
		}else if (src_state == 19){
		}
	}
#endif

#ifdef	Synthetic_simple_100states_kernel
	for(src_state = 0; src_state < 100; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,0,1);
			/***dest 3***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,3,mask,0,0,1);
			/***dest 4***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,4,mask,0,0,1);
			/***dest 5***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,5,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 6***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 8***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 9***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 10***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,0,1);
			/***dest 11***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,11,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 12***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,12,mask,0,0,1);
		}else if (src_state == 7){
			/***dest 13***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,13,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 14***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,14,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 15***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,15,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 16***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,16,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 5***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,5,mask,0,0,1);
			/***dest 17***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,17,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 18***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,18,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 19***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,19,mask,0,0,1);
		}else if (src_state == 14){
			/***dest 20***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,20,mask,0,0,1);
		}else if (src_state == 15){
			/***dest 9***/
			mask = match_check(s_char,21,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,9,mask,0,0,1);
			/***dest 21***/
			mask = match_check(s_char,22,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,21,mask,0,0,1);
		}else if (src_state == 16){
			/***dest 22***/
			mask = match_check(s_char,23,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,22,mask,0,0,1);
		}else if (src_state == 17){
			/***dest 23***/
			mask = match_check(s_char,24,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,23,mask,0,0,1);
		}else if (src_state == 18){
			/***dest 24***/
			mask = match_check(s_char,25,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,24,mask,0,0,1);
		}else if (src_state == 19){
			/***dest 25***/
			mask = match_check(s_char,26,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,25,mask,0,0,1);
		}else if (src_state == 20){
			/***dest 26***/
			mask = match_check(s_char,27,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,26,mask,0,0,1);
		}else if (src_state == 21){
			/***dest 27***/
			mask = match_check(s_char,28,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,21,27,mask,0,0,1);
		}else if (src_state == 22){
			/***dest 28***/
			mask = match_check(s_char,29,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,28,mask,0,0,1);
		}else if (src_state == 23){
			/***dest 29***/
			mask = match_check(s_char,30,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,23,29,mask,0,0,1);
		}else if (src_state == 24){
			/***dest 30***/
			mask = match_check(s_char,31,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,30,mask,0,0,1);
		}else if (src_state == 25){
			/***dest 31***/
			mask = match_check(s_char,32,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,25,31,mask,0,0,1);
		}else if (src_state == 26){
			/***dest 32***/
			mask = match_check(s_char,33,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,32,mask,0,0,1);
		}else if (src_state == 27){
			/***dest 33***/
			mask = match_check(s_char,34,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,27,33,mask,0,0,1);
		}else if (src_state == 28){
			/***dest 34***/
			mask = match_check(s_char,35,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,34,mask,0,0,1);
		}else if (src_state == 29){
			/***dest 35***/
			mask = match_check(s_char,36,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,35,mask,0,0,1);
		}else if (src_state == 30){
			/***dest 36***/
			mask = match_check(s_char,37,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,36,mask,0,0,1);
		}else if (src_state == 31){
			/***dest 37***/
			mask = match_check(s_char,38,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,37,mask,0,0,1);
		}else if (src_state == 32){
			/***dest 38***/
			mask = match_check(s_char,39,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,32,38,mask,0,0,1);
		}else if (src_state == 33){
			/***dest 39***/
			mask = match_check(s_char,40,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,33,39,mask,0,0,1);
		}else if (src_state == 34){
			/***dest 40***/
			mask = match_check(s_char,41,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,34,40,mask,0,0,1);
		}else if (src_state == 35){
			/***dest 41***/
			mask = match_check(s_char,42,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,41,mask,0,0,1);
			/***dest 42***/
			mask = match_check(s_char,43,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,42,mask,0,0,1);
			/***dest 43***/
			mask = match_check(s_char,44,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,43,mask,0,0,1);
			/***dest 44***/
			mask = match_check(s_char,45,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,44,mask,0,0,1);
			/***dest 45***/
			mask = match_check(s_char,46,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,45,mask,0,0,1);
		}else if (src_state == 36){
			/***dest 46***/
			mask = match_check(s_char,47,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,36,46,mask,0,0,1);
		}else if (src_state == 37){
			/***dest 31***/
			mask = match_check(s_char,48,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,31,mask,0,0,1);
			/***dest 47***/
			mask = match_check(s_char,49,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,47,mask,0,0,1);
		}else if (src_state == 38){
			/***dest 48***/
			mask = match_check(s_char,50,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,38,48,mask,0,0,1);
		}else if (src_state == 39){
			/***dest 49***/
			mask = match_check(s_char,51,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,49,mask,0,0,1);
		}else if (src_state == 40){
			/***dest 50***/
			mask = match_check(s_char,52,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,50,mask,0,0,1);
		}else if (src_state == 41){
			/***dest 51***/
			mask = match_check(s_char,53,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,51,mask,0,0,1);
		}else if (src_state == 42){
			/***dest 52***/
			mask = match_check(s_char,54,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,52,mask,0,0,1);
		}else if (src_state == 43){
			/***dest 53***/
			mask = match_check(s_char,55,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,53,mask,0,0,1);
		}else if (src_state == 44){
			/***dest 54***/
			mask = match_check(s_char,56,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,54,mask,0,0,1);
		}else if (src_state == 45){
			/***dest 55***/
			mask = match_check(s_char,57,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,55,mask,0,0,1);
		}else if (src_state == 46){
			/***dest 56***/
			mask = match_check(s_char,58,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,56,mask,0,0,1);
		}else if (src_state == 47){
			/***dest 57***/
			mask = match_check(s_char,59,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,57,mask,0,0,1);
		}else if (src_state == 48){
			/***dest 58***/
			mask = match_check(s_char,60,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,58,mask,0,0,1);
		}else if (src_state == 49){
			/***dest 59***/
			mask = match_check(s_char,61,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,49,59,mask,0,0,1);
		}else if (src_state == 50){
			/***dest 60***/
			mask = match_check(s_char,62,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,60,mask,0,0,1);
		}else if (src_state == 51){
			/***dest 61***/
			mask = match_check(s_char,63,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,61,mask,0,0,1);
		}else if (src_state == 52){
			/***dest 62***/
			mask = match_check(s_char,64,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,62,mask,0,0,1);
		}else if (src_state == 53){
			/***dest 63***/
			mask = match_check(s_char,65,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,63,mask,0,0,1);
		}else if (src_state == 54){
			/***dest 64***/
			mask = match_check(s_char,66,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,64,mask,0,0,1);
		}else if (src_state == 55){
			/***dest 65***/
			mask = match_check(s_char,67,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,65,mask,0,0,1);
		}else if (src_state == 56){
			/***dest 46***/
			mask = match_check(s_char,68,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,46,mask,0,0,1);
			/***dest 66***/
			mask = match_check(s_char,69,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,66,mask,0,0,1);
		}else if (src_state == 57){
			/***dest 67***/
			mask = match_check(s_char,70,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,67,mask,0,0,1);
		}else if (src_state == 58){
			/***dest 68***/
			mask = match_check(s_char,71,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,68,mask,0,0,1);
		}else if (src_state == 59){
			/***dest 69***/
			mask = match_check(s_char,72,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,59,69,mask,0,0,1);
		}else if (src_state == 60){
			/***dest 70***/
			mask = match_check(s_char,73,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,60,70,mask,0,0,1);
		}else if (src_state == 61){
			/***dest 71***/
			mask = match_check(s_char,74,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,61,71,mask,0,0,1);
		}else if (src_state == 62){
			/***dest 72***/
			mask = match_check(s_char,75,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,72,mask,0,0,1);
		}else if (src_state == 63){
			/***dest 73***/
			mask = match_check(s_char,76,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,73,mask,0,0,1);
		}else if (src_state == 64){
			/***dest 74***/
			mask = match_check(s_char,77,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,74,mask,0,0,1);
		}else if (src_state == 65){
			/***dest 75***/
			mask = match_check(s_char,78,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,65,75,mask,0,0,1);
		}else if (src_state == 66){
			/***dest 76***/
			mask = match_check(s_char,79,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,66,76,mask,0,0,1);
		}else if (src_state == 67){
			/***dest 77***/
			mask = match_check(s_char,80,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,77,mask,0,0,1);
		}else if (src_state == 68){
			/***dest 78***/
			mask = match_check(s_char,81,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,68,78,mask,0,0,1);
		}else if (src_state == 69){
			/***dest 79***/
			mask = match_check(s_char,82,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,69,79,mask,0,0,1);
		}else if (src_state == 70){
			/***dest 80***/
			mask = match_check(s_char,83,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,80,mask,0,0,1);
		}else if (src_state == 71){
			/***dest 81***/
			mask = match_check(s_char,84,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,71,81,mask,0,0,1);
		}else if (src_state == 72){
			/***dest 82***/
			mask = match_check(s_char,85,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,72,82,mask,0,0,1);
		}else if (src_state == 73){
			/***dest 83***/
			mask = match_check(s_char,86,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,73,83,mask,0,0,1);
		}else if (src_state == 74){
			/***dest 84***/
			mask = match_check(s_char,87,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,74,84,mask,0,0,1);
		}else if (src_state == 75){
			/***dest 85***/
			mask = match_check(s_char,88,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,75,85,mask,0,0,1);
		}else if (src_state == 76){
			/***dest 86***/
			mask = match_check(s_char,89,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,76,86,mask,0,0,1);
		}else if (src_state == 77){
			/***dest 87***/
			mask = match_check(s_char,90,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,77,87,mask,0,0,1);
		}else if (src_state == 78){
			/***dest 88***/
			mask = match_check(s_char,91,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,78,88,mask,0,0,1);
		}else if (src_state == 79){
			/***dest 89***/
			mask = match_check(s_char,92,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,79,89,mask,0,0,1);
		}else if (src_state == 80){
			/***dest 90***/
			mask = match_check(s_char,93,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,80,90,mask,0,0,1);
		}else if (src_state == 81){
			/***dest 91***/
			mask = match_check(s_char,94,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,81,91,mask,0,0,1);
		}else if (src_state == 82){
			/***dest 92***/
			mask = match_check(s_char,95,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,82,92,mask,0,0,1);
		}else if (src_state == 83){
			/***dest 93***/
			mask = match_check(s_char,96,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,83,93,mask,0,0,1);
		}else if (src_state == 84){
			/***dest 94***/
			mask = match_check(s_char,97,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,84,94,mask,0,0,1);
		}else if (src_state == 85){
			/***dest 95***/
			mask = match_check(s_char,98,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,85,95,mask,0,0,1);
		}else if (src_state == 86){
			/***dest 96***/
			mask = match_check(s_char,99,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,86,96,mask,0,0,1);
		}else if (src_state == 87){
			/***dest 97***/
			mask = match_check(s_char,100,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,87,97,mask,0,0,1);
		}else if (src_state == 88){
			/***dest 98***/
			mask = match_check(s_char,101,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,88,98,mask,0,0,1);
		}else if (src_state == 89){
			/***dest 99***/
			mask = match_check(s_char,102,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,89,99,mask,0,0,1);
		}else if (src_state == 90){
		}else if (src_state == 91){
		}else if (src_state == 92){
		}else if (src_state == 93){
		}else if (src_state == 94){
			/***dest 84***/
			mask = match_check(s_char,103,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,94,84,mask,0,0,1);
		}else if (src_state == 95){
		}else if (src_state == 96){
		}else if (src_state == 97){
		}else if (src_state == 98){
		}else if (src_state == 99){
		}
	}
#endif

#ifdef	Synthetic_simple_with_selfloop_20states_kernel
	for(src_state = 0; src_state < 20; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,0,1);
			/***dest 3***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,3,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 1***/
			character_transitions_update(cfBV, ctof,1,1,mask,1,0,0);
			/***dest 4***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,4,mask,0,0,1);
			/***dest 5***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,5,mask,0,0,1);
			/***dest 6***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 0***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,0,mask,0,0,1);
			/***dest 8***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 4***/
			character_transitions_update(cfBV, ctof,4,4,mask,1,0,0);
			/***dest 9***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 10***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 11***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,11,mask,0,0,1);
		}else if (src_state == 7){
			/***dest 7***/
			character_transitions_update(cfBV, ctof,7,7,mask,1,0,0);
			/***dest 12***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,12,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 8***/
			character_transitions_update(cfBV, ctof,8,8,mask,1,0,0);
			/***dest 13***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,13,mask,0,0,1);
			/***dest 14***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,14,mask,0,0,1);
			/***dest 15***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,15,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 16***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,16,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 17***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,17,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 18***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,18,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 19***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,19,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 8***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,8,mask,0,0,1);
		}else if (src_state == 14){
		}else if (src_state == 15){
		}else if (src_state == 16){
		}else if (src_state == 17){
		}else if (src_state == 18){
		}else if (src_state == 19){
		}
	}
#endif

#ifdef	Synthetic_simple_with_selfloop_100states_kernel
	for(src_state = 0; src_state < 100; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,0,1);
			/***dest 3***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,3,mask,0,0,1);
			/***dest 4***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,4,mask,0,0,1);
			/***dest 5***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,5,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 1***/
			character_transitions_update(cfBV, ctof,1,1,mask,1,0,0);
			/***dest 6***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 8***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 4***/
			character_transitions_update(cfBV, ctof,4,4,mask,1,0,0);
			/***dest 9***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 5***/
			character_transitions_update(cfBV, ctof,5,5,mask,1,0,0);
			/***dest 10***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,0,1);
			/***dest 11***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,11,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 12***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,12,mask,0,0,1);
		}else if (src_state == 7){
			/***dest 7***/
			character_transitions_update(cfBV, ctof,7,7,mask,1,0,0);
			/***dest 13***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,13,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 14***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,14,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 15***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,15,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 16***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,16,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 5***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,5,mask,0,0,1);
			/***dest 17***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,17,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 18***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,18,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 19***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,19,mask,0,0,1);
		}else if (src_state == 14){
			/***dest 14***/
			character_transitions_update(cfBV, ctof,14,14,mask,1,0,0);
			/***dest 20***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,20,mask,0,0,1);
		}else if (src_state == 15){
			/***dest 9***/
			mask = match_check(s_char,21,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,9,mask,0,0,1);
			/***dest 21***/
			mask = match_check(s_char,22,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,21,mask,0,0,1);
		}else if (src_state == 16){
			/***dest 22***/
			mask = match_check(s_char,23,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,22,mask,0,0,1);
		}else if (src_state == 17){
			/***dest 23***/
			mask = match_check(s_char,24,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,23,mask,0,0,1);
		}else if (src_state == 18){
			/***dest 18***/
			character_transitions_update(cfBV, ctof,18,18,mask,1,0,0);
			/***dest 24***/
			mask = match_check(s_char,25,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,24,mask,0,0,1);
		}else if (src_state == 19){
			/***dest 25***/
			mask = match_check(s_char,26,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,25,mask,0,0,1);
		}else if (src_state == 20){
			/***dest 26***/
			mask = match_check(s_char,27,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,26,mask,0,0,1);
		}else if (src_state == 21){
			/***dest 21***/
			character_transitions_update(cfBV, ctof,21,21,mask,1,0,0);
			/***dest 27***/
			mask = match_check(s_char,28,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,21,27,mask,0,0,1);
		}else if (src_state == 22){
			/***dest 22***/
			character_transitions_update(cfBV, ctof,22,22,mask,1,0,0);
			/***dest 28***/
			mask = match_check(s_char,29,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,28,mask,0,0,1);
		}else if (src_state == 23){
			/***dest 29***/
			mask = match_check(s_char,30,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,23,29,mask,0,0,1);
		}else if (src_state == 24){
			/***dest 30***/
			mask = match_check(s_char,31,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,30,mask,0,0,1);
		}else if (src_state == 25){
			/***dest 25***/
			character_transitions_update(cfBV, ctof,25,25,mask,1,0,0);
			/***dest 31***/
			mask = match_check(s_char,32,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,25,31,mask,0,0,1);
		}else if (src_state == 26){
			/***dest 32***/
			mask = match_check(s_char,33,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,32,mask,0,0,1);
		}else if (src_state == 27){
			/***dest 33***/
			mask = match_check(s_char,34,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,27,33,mask,0,0,1);
		}else if (src_state == 28){
			/***dest 34***/
			mask = match_check(s_char,35,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,34,mask,0,0,1);
		}else if (src_state == 29){
			/***dest 35***/
			mask = match_check(s_char,36,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,35,mask,0,0,1);
		}else if (src_state == 30){
			/***dest 36***/
			mask = match_check(s_char,37,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,36,mask,0,0,1);
		}else if (src_state == 31){
			/***dest 37***/
			mask = match_check(s_char,38,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,37,mask,0,0,1);
		}else if (src_state == 32){
			/***dest 32***/
			character_transitions_update(cfBV, ctof,32,32,mask,1,0,0);
			/***dest 38***/
			mask = match_check(s_char,39,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,32,38,mask,0,0,1);
		}else if (src_state == 33){
			/***dest 39***/
			mask = match_check(s_char,40,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,33,39,mask,0,0,1);
		}else if (src_state == 34){
			/***dest 40***/
			mask = match_check(s_char,41,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,34,40,mask,0,0,1);
		}else if (src_state == 35){
			/***dest 35***/
			character_transitions_update(cfBV, ctof,35,35,mask,1,0,0);
			/***dest 41***/
			mask = match_check(s_char,42,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,41,mask,0,0,1);
			/***dest 42***/
			mask = match_check(s_char,43,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,42,mask,0,0,1);
			/***dest 43***/
			mask = match_check(s_char,44,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,43,mask,0,0,1);
			/***dest 44***/
			mask = match_check(s_char,45,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,44,mask,0,0,1);
			/***dest 45***/
			mask = match_check(s_char,46,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,45,mask,0,0,1);
		}else if (src_state == 36){
			/***dest 46***/
			mask = match_check(s_char,47,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,36,46,mask,0,0,1);
		}else if (src_state == 37){
			/***dest 31***/
			mask = match_check(s_char,48,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,31,mask,0,0,1);
			/***dest 47***/
			mask = match_check(s_char,49,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,47,mask,0,0,1);
		}else if (src_state == 38){
			/***dest 48***/
			mask = match_check(s_char,50,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,38,48,mask,0,0,1);
		}else if (src_state == 39){
			/***dest 39***/
			character_transitions_update(cfBV, ctof,39,39,mask,1,0,0);
			/***dest 49***/
			mask = match_check(s_char,51,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,49,mask,0,0,1);
		}else if (src_state == 40){
			/***dest 50***/
			mask = match_check(s_char,52,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,50,mask,0,0,1);
		}else if (src_state == 41){
			/***dest 41***/
			character_transitions_update(cfBV, ctof,41,41,mask,1,0,0);
			/***dest 51***/
			mask = match_check(s_char,53,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,51,mask,0,0,1);
		}else if (src_state == 42){
			/***dest 52***/
			mask = match_check(s_char,54,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,52,mask,0,0,1);
		}else if (src_state == 43){
			/***dest 53***/
			mask = match_check(s_char,55,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,53,mask,0,0,1);
		}else if (src_state == 44){
			/***dest 54***/
			mask = match_check(s_char,56,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,54,mask,0,0,1);
		}else if (src_state == 45){
			/***dest 55***/
			mask = match_check(s_char,57,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,55,mask,0,0,1);
		}else if (src_state == 46){
			/***dest 46***/
			character_transitions_update(cfBV, ctof,46,46,mask,1,0,0);
			/***dest 56***/
			mask = match_check(s_char,58,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,56,mask,0,0,1);
		}else if (src_state == 47){
			/***dest 57***/
			mask = match_check(s_char,59,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,57,mask,0,0,1);
		}else if (src_state == 48){
			/***dest 58***/
			mask = match_check(s_char,60,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,58,mask,0,0,1);
		}else if (src_state == 49){
			/***dest 59***/
			mask = match_check(s_char,61,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,49,59,mask,0,0,1);
		}else if (src_state == 50){
			/***dest 50***/
			character_transitions_update(cfBV, ctof,50,50,mask,1,0,0);
			/***dest 60***/
			mask = match_check(s_char,62,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,60,mask,0,0,1);
		}else if (src_state == 51){
			/***dest 61***/
			mask = match_check(s_char,63,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,61,mask,0,0,1);
		}else if (src_state == 52){
			/***dest 52***/
			character_transitions_update(cfBV, ctof,52,52,mask,1,0,0);
			/***dest 62***/
			mask = match_check(s_char,64,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,62,mask,0,0,1);
		}else if (src_state == 53){
			/***dest 63***/
			mask = match_check(s_char,65,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,63,mask,0,0,1);
		}else if (src_state == 54){
			/***dest 64***/
			mask = match_check(s_char,66,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,64,mask,0,0,1);
		}else if (src_state == 55){
			/***dest 55***/
			character_transitions_update(cfBV, ctof,55,55,mask,1,0,0);
			/***dest 65***/
			mask = match_check(s_char,67,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,65,mask,0,0,1);
		}else if (src_state == 56){
			/***dest 46***/
			mask = match_check(s_char,68,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,46,mask,0,0,1);
			/***dest 66***/
			mask = match_check(s_char,69,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,66,mask,0,0,1);
		}else if (src_state == 57){
			/***dest 67***/
			mask = match_check(s_char,70,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,67,mask,0,0,1);
		}else if (src_state == 58){
			/***dest 58***/
			character_transitions_update(cfBV, ctof,58,58,mask,1,0,0);
			/***dest 68***/
			mask = match_check(s_char,71,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,68,mask,0,0,1);
		}else if (src_state == 59){
			/***dest 69***/
			mask = match_check(s_char,72,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,59,69,mask,0,0,1);
		}else if (src_state == 60){
			/***dest 70***/
			mask = match_check(s_char,73,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,60,70,mask,0,0,1);
		}else if (src_state == 61){
			/***dest 71***/
			mask = match_check(s_char,74,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,61,71,mask,0,0,1);
		}else if (src_state == 62){
			/***dest 72***/
			mask = match_check(s_char,75,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,72,mask,0,0,1);
		}else if (src_state == 63){
			/***dest 73***/
			mask = match_check(s_char,76,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,73,mask,0,0,1);
		}else if (src_state == 64){
			/***dest 64***/
			character_transitions_update(cfBV, ctof,64,64,mask,1,0,0);
			/***dest 74***/
			mask = match_check(s_char,77,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,74,mask,0,0,1);
		}else if (src_state == 65){
			/***dest 75***/
			mask = match_check(s_char,78,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,65,75,mask,0,0,1);
		}else if (src_state == 66){
			/***dest 76***/
			mask = match_check(s_char,79,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,66,76,mask,0,0,1);
		}else if (src_state == 67){
			/***dest 67***/
			character_transitions_update(cfBV, ctof,67,67,mask,1,0,0);
			/***dest 77***/
			mask = match_check(s_char,80,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,77,mask,0,0,1);
		}else if (src_state == 68){
			/***dest 78***/
			mask = match_check(s_char,81,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,68,78,mask,0,0,1);
		}else if (src_state == 69){
			/***dest 79***/
			mask = match_check(s_char,82,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,69,79,mask,0,0,1);
		}else if (src_state == 70){
			/***dest 70***/
			character_transitions_update(cfBV, ctof,70,70,mask,1,0,0);
			/***dest 80***/
			mask = match_check(s_char,83,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,80,mask,0,0,1);
		}else if (src_state == 71){
			/***dest 71***/
			character_transitions_update(cfBV, ctof,71,71,mask,1,0,0);
			/***dest 81***/
			mask = match_check(s_char,84,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,71,81,mask,0,0,1);
		}else if (src_state == 72){
			/***dest 82***/
			mask = match_check(s_char,85,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,72,82,mask,0,0,1);
		}else if (src_state == 73){
			/***dest 73***/
			character_transitions_update(cfBV, ctof,73,73,mask,1,0,0);
			/***dest 83***/
			mask = match_check(s_char,86,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,73,83,mask,0,0,1);
		}else if (src_state == 74){
			/***dest 84***/
			mask = match_check(s_char,87,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,74,84,mask,0,0,1);
		}else if (src_state == 75){
			/***dest 75***/
			character_transitions_update(cfBV, ctof,75,75,mask,1,0,0);
			/***dest 85***/
			mask = match_check(s_char,88,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,75,85,mask,0,0,1);
		}else if (src_state == 76){
			/***dest 86***/
			mask = match_check(s_char,89,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,76,86,mask,0,0,1);
		}else if (src_state == 77){
			/***dest 87***/
			mask = match_check(s_char,90,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,77,87,mask,0,0,1);
		}else if (src_state == 78){
			/***dest 88***/
			mask = match_check(s_char,91,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,78,88,mask,0,0,1);
		}else if (src_state == 79){
			/***dest 89***/
			mask = match_check(s_char,92,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,79,89,mask,0,0,1);
		}else if (src_state == 80){
			/***dest 90***/
			mask = match_check(s_char,93,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,80,90,mask,0,0,1);
		}else if (src_state == 81){
			/***dest 91***/
			mask = match_check(s_char,94,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,81,91,mask,0,0,1);
		}else if (src_state == 82){
			/***dest 92***/
			mask = match_check(s_char,95,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,82,92,mask,0,0,1);
		}else if (src_state == 83){
			/***dest 93***/
			mask = match_check(s_char,96,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,83,93,mask,0,0,1);
		}else if (src_state == 84){
			/***dest 94***/
			mask = match_check(s_char,97,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,84,94,mask,0,0,1);
		}else if (src_state == 85){
			/***dest 95***/
			mask = match_check(s_char,98,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,85,95,mask,0,0,1);
		}else if (src_state == 86){
			/***dest 96***/
			mask = match_check(s_char,99,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,86,96,mask,0,0,1);
		}else if (src_state == 87){
			/***dest 97***/
			mask = match_check(s_char,100,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,87,97,mask,0,0,1);
		}else if (src_state == 88){
			/***dest 98***/
			mask = match_check(s_char,101,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,88,98,mask,0,0,1);
		}else if (src_state == 89){
			/***dest 99***/
			mask = match_check(s_char,102,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,89,99,mask,0,0,1);
		}else if (src_state == 90){
		}else if (src_state == 91){
		}else if (src_state == 92){
		}else if (src_state == 93){
		}else if (src_state == 94){
			/***dest 84***/
			mask = match_check(s_char,103,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,94,84,mask,0,0,1);
		}else if (src_state == 95){
		}else if (src_state == 96){
		}else if (src_state == 97){
		}else if (src_state == 98){
		}else if (src_state == 99){
		}
	}
#endif

#ifdef	Synthetic_simple_with_wildcard_20states_kernel
	for(src_state = 0; src_state < 20; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			character_transitions_update(cfBV, ctof,0,2,mask,1,0,0);
			/***dest 3***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,3,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 4***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,4,mask,0,0,1);
			/***dest 5***/
			character_transitions_update(cfBV, ctof,1,5,mask,1,0,0);
			/***dest 6***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 0***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,0,mask,0,0,1);
			/***dest 8***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 9***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 10***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 11***/
			character_transitions_update(cfBV, ctof,6,11,mask,1,0,0);
		}else if (src_state == 7){
			/***dest 12***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,12,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 13***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,13,mask,0,0,1);
			/***dest 14***/
			character_transitions_update(cfBV, ctof,8,14,mask,1,0,0);
			/***dest 15***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,15,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 16***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,16,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 17***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,17,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 18***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,18,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 19***/
			character_transitions_update(cfBV, ctof,12,19,mask,1,0,0);
		}else if (src_state == 13){
			/***dest 8***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,8,mask,0,0,1);
		}else if (src_state == 14){
		}else if (src_state == 15){
		}else if (src_state == 16){
		}else if (src_state == 17){
		}else if (src_state == 18){
		}else if (src_state == 19){
		}
	}
#endif

#ifdef	Synthetic_simple_with_wildcard_100states_kernel
	for(src_state = 0; src_state < 100; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,0,1);
			/***dest 3***/
			character_transitions_update(cfBV, ctof,0,3,mask,1,0,0);
			/***dest 4***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,4,mask,0,0,1);
			/***dest 5***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,5,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 6***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 8***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 9***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,0,1);
		}else if (src_state == 5){
			/***dest 10***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,0,1);
			/***dest 11***/
			character_transitions_update(cfBV, ctof,5,11,mask,1,0,0);
		}else if (src_state == 6){
			/***dest 12***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,12,mask,0,0,1);
		}else if (src_state == 7){
			/***dest 13***/
			character_transitions_update(cfBV, ctof,7,13,mask,1,0,0);
		}else if (src_state == 8){
			/***dest 14***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,14,mask,0,0,1);
		}else if (src_state == 9){
			/***dest 15***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,15,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 16***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,16,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 5***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,5,mask,0,0,1);
			/***dest 17***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,17,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 18***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,18,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 19***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,19,mask,0,0,1);
		}else if (src_state == 14){
			/***dest 20***/
			character_transitions_update(cfBV, ctof,14,20,mask,1,0,0);
		}else if (src_state == 15){
			/***dest 9***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,9,mask,0,0,1);
			/***dest 21***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,21,mask,0,0,1);
		}else if (src_state == 16){
			/***dest 22***/
			character_transitions_update(cfBV, ctof,16,22,mask,1,0,0);
		}else if (src_state == 17){
			/***dest 23***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,23,mask,0,0,1);
		}else if (src_state == 18){
			/***dest 24***/
			character_transitions_update(cfBV, ctof,18,24,mask,1,0,0);
		}else if (src_state == 19){
			/***dest 25***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,25,mask,0,0,1);
		}else if (src_state == 20){
			/***dest 26***/
			mask = match_check(s_char,21,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,26,mask,0,0,1);
		}else if (src_state == 21){
			/***dest 27***/
			mask = match_check(s_char,22,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,21,27,mask,0,0,1);
		}else if (src_state == 22){
			/***dest 28***/
			mask = match_check(s_char,23,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,28,mask,0,0,1);
		}else if (src_state == 23){
			/***dest 29***/
			character_transitions_update(cfBV, ctof,23,29,mask,1,0,0);
		}else if (src_state == 24){
			/***dest 30***/
			mask = match_check(s_char,24,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,30,mask,0,0,1);
		}else if (src_state == 25){
			/***dest 31***/
			character_transitions_update(cfBV, ctof,25,31,mask,1,0,0);
		}else if (src_state == 26){
			/***dest 32***/
			mask = match_check(s_char,25,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,32,mask,0,0,1);
		}else if (src_state == 27){
			/***dest 33***/
			character_transitions_update(cfBV, ctof,27,33,mask,1,0,0);
		}else if (src_state == 28){
			/***dest 34***/
			mask = match_check(s_char,26,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,34,mask,0,0,1);
		}else if (src_state == 29){
			/***dest 35***/
			mask = match_check(s_char,27,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,35,mask,0,0,1);
		}else if (src_state == 30){
			/***dest 36***/
			mask = match_check(s_char,28,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,36,mask,0,0,1);
		}else if (src_state == 31){
			/***dest 37***/
			mask = match_check(s_char,29,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,37,mask,0,0,1);
		}else if (src_state == 32){
			/***dest 38***/
			character_transitions_update(cfBV, ctof,32,38,mask,1,0,0);
		}else if (src_state == 33){
			/***dest 39***/
			mask = match_check(s_char,30,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,33,39,mask,0,0,1);
		}else if (src_state == 34){
			/***dest 40***/
			character_transitions_update(cfBV, ctof,34,40,mask,1,0,0);
		}else if (src_state == 35){
			/***dest 41***/
			character_transitions_update(cfBV, ctof,35,41,mask,1,0,0);
			/***dest 42***/
			character_transitions_update(cfBV, ctof,35,42,mask,1,0,0);
			/***dest 43***/
			mask = match_check(s_char,31,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,43,mask,0,0,1);
			/***dest 44***/
			mask = match_check(s_char,32,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,44,mask,0,0,1);
			/***dest 45***/
			mask = match_check(s_char,33,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,45,mask,0,0,1);
		}else if (src_state == 36){
			/***dest 46***/
			character_transitions_update(cfBV, ctof,36,46,mask,1,0,0);
		}else if (src_state == 37){
			/***dest 31***/
			mask = match_check(s_char,34,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,31,mask,0,0,1);
			/***dest 47***/
			mask = match_check(s_char,35,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,47,mask,0,0,1);
		}else if (src_state == 38){
			/***dest 48***/
			mask = match_check(s_char,36,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,38,48,mask,0,0,1);
		}else if (src_state == 39){
			/***dest 49***/
			mask = match_check(s_char,37,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,49,mask,0,0,1);
		}else if (src_state == 40){
			/***dest 50***/
			mask = match_check(s_char,38,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,50,mask,0,0,1);
		}else if (src_state == 41){
			/***dest 51***/
			mask = match_check(s_char,39,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,51,mask,0,0,1);
		}else if (src_state == 42){
			/***dest 52***/
			mask = match_check(s_char,40,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,52,mask,0,0,1);
		}else if (src_state == 43){
			/***dest 53***/
			character_transitions_update(cfBV, ctof,43,53,mask,1,0,0);
		}else if (src_state == 44){
			/***dest 54***/
			mask = match_check(s_char,41,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,54,mask,0,0,1);
		}else if (src_state == 45){
			/***dest 55***/
			mask = match_check(s_char,42,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,55,mask,0,0,1);
		}else if (src_state == 46){
			/***dest 56***/
			mask = match_check(s_char,43,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,56,mask,0,0,1);
		}else if (src_state == 47){
			/***dest 57***/
			character_transitions_update(cfBV, ctof,47,57,mask,1,0,0);
		}else if (src_state == 48){
			/***dest 58***/
			mask = match_check(s_char,44,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,58,mask,0,0,1);
		}else if (src_state == 49){
			/***dest 59***/
			character_transitions_update(cfBV, ctof,49,59,mask,1,0,0);
		}else if (src_state == 50){
			/***dest 60***/
			mask = match_check(s_char,45,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,60,mask,0,0,1);
		}else if (src_state == 51){
			/***dest 61***/
			mask = match_check(s_char,46,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,61,mask,0,0,1);
		}else if (src_state == 52){
			/***dest 62***/
			mask = match_check(s_char,47,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,62,mask,0,0,1);
		}else if (src_state == 53){
			/***dest 63***/
			mask = match_check(s_char,48,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,63,mask,0,0,1);
		}else if (src_state == 54){
			/***dest 64***/
			mask = match_check(s_char,49,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,64,mask,0,0,1);
		}else if (src_state == 55){
			/***dest 65***/
			mask = match_check(s_char,50,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,65,mask,0,0,1);
		}else if (src_state == 56){
			/***dest 46***/
			mask = match_check(s_char,51,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,46,mask,0,0,1);
			/***dest 66***/
			mask = match_check(s_char,52,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,66,mask,0,0,1);
		}else if (src_state == 57){
			/***dest 67***/
			mask = match_check(s_char,53,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,67,mask,0,0,1);
		}else if (src_state == 58){
			/***dest 68***/
			mask = match_check(s_char,54,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,68,mask,0,0,1);
		}else if (src_state == 59){
			/***dest 69***/
			mask = match_check(s_char,55,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,59,69,mask,0,0,1);
		}else if (src_state == 60){
			/***dest 70***/
			character_transitions_update(cfBV, ctof,60,70,mask,1,0,0);
		}else if (src_state == 61){
			/***dest 71***/
			mask = match_check(s_char,56,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,61,71,mask,0,0,1);
		}else if (src_state == 62){
			/***dest 72***/
			character_transitions_update(cfBV, ctof,62,72,mask,1,0,0);
		}else if (src_state == 63){
			/***dest 73***/
			mask = match_check(s_char,57,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,73,mask,0,0,1);
		}else if (src_state == 64){
			/***dest 74***/
			character_transitions_update(cfBV, ctof,64,74,mask,1,0,0);
		}else if (src_state == 65){
			/***dest 75***/
			character_transitions_update(cfBV, ctof,65,75,mask,1,0,0);
		}else if (src_state == 66){
			/***dest 76***/
			character_transitions_update(cfBV, ctof,66,76,mask,1,0,0);
		}else if (src_state == 67){
			/***dest 77***/
			mask = match_check(s_char,58,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,77,mask,0,0,1);
		}else if (src_state == 68){
			/***dest 78***/
			character_transitions_update(cfBV, ctof,68,78,mask,1,0,0);
		}else if (src_state == 69){
			/***dest 79***/
			mask = match_check(s_char,59,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,69,79,mask,0,0,1);
		}else if (src_state == 70){
			/***dest 80***/
			mask = match_check(s_char,60,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,80,mask,0,0,1);
		}else if (src_state == 71){
			/***dest 81***/
			mask = match_check(s_char,61,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,71,81,mask,0,0,1);
		}else if (src_state == 72){
			/***dest 82***/
			mask = match_check(s_char,62,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,72,82,mask,0,0,1);
		}else if (src_state == 73){
			/***dest 83***/
			mask = match_check(s_char,63,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,73,83,mask,0,0,1);
		}else if (src_state == 74){
			/***dest 84***/
			mask = match_check(s_char,64,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,74,84,mask,0,0,1);
		}else if (src_state == 75){
			/***dest 85***/
			mask = match_check(s_char,65,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,75,85,mask,0,0,1);
		}else if (src_state == 76){
			/***dest 86***/
			mask = match_check(s_char,66,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,76,86,mask,0,0,1);
		}else if (src_state == 77){
			/***dest 87***/
			character_transitions_update(cfBV, ctof,77,87,mask,1,0,0);
		}else if (src_state == 78){
			/***dest 88***/
			mask = match_check(s_char,67,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,78,88,mask,0,0,1);
		}else if (src_state == 79){
			/***dest 89***/
			character_transitions_update(cfBV, ctof,79,89,mask,1,0,0);
		}else if (src_state == 80){
			/***dest 90***/
			mask = match_check(s_char,68,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,80,90,mask,0,0,1);
		}else if (src_state == 81){
			/***dest 91***/
			mask = match_check(s_char,69,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,81,91,mask,0,0,1);
		}else if (src_state == 82){
			/***dest 92***/
			mask = match_check(s_char,70,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,82,92,mask,0,0,1);
		}else if (src_state == 83){
			/***dest 93***/
			mask = match_check(s_char,71,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,83,93,mask,0,0,1);
		}else if (src_state == 84){
			/***dest 94***/
			mask = match_check(s_char,72,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,84,94,mask,0,0,1);
		}else if (src_state == 85){
			/***dest 95***/
			mask = match_check(s_char,73,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,85,95,mask,0,0,1);
		}else if (src_state == 86){
			/***dest 96***/
			mask = match_check(s_char,74,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,86,96,mask,0,0,1);
		}else if (src_state == 87){
			/***dest 97***/
			mask = match_check(s_char,75,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,87,97,mask,0,0,1);
		}else if (src_state == 88){
			/***dest 98***/
			mask = match_check(s_char,76,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,88,98,mask,0,0,1);
		}else if (src_state == 89){
			/***dest 99***/
			mask = match_check(s_char,77,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,89,99,mask,0,0,1);
		}else if (src_state == 90){
		}else if (src_state == 91){
		}else if (src_state == 92){
		}else if (src_state == 93){
		}else if (src_state == 94){
			/***dest 84***/
			mask = match_check(s_char,78,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,94,84,mask,0,0,1);
		}else if (src_state == 95){
		}else if (src_state == 96){
		}else if (src_state == 97){
		}else if (src_state == 98){
		}else if (src_state == 99){
		}
	}
#endif

#ifdef	Synthetic_simple_with_neg_20states_kernel
	for(src_state = 0; src_state < 20; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,1,0);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,0,1);
			/***dest 3***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,3,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 4***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,4,mask,0,1,0);
			/***dest 5***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,5,mask,0,0,1);
			/***dest 6***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 0***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,0,mask,0,0,1);
			/***dest 8***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 9***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,1,0);
		}else if (src_state == 5){
			/***dest 10***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 11***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,11,mask,0,0,1);
		}else if (src_state == 7){
			/***dest 12***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,12,mask,0,1,0);
		}else if (src_state == 8){
			/***dest 13***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,13,mask,0,0,1);
			/***dest 14***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,14,mask,0,0,1);
			/***dest 15***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,15,mask,0,1,0);
		}else if (src_state == 9){
			/***dest 16***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,16,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 17***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,17,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 18***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,18,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 19***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,19,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 8***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,8,mask,0,0,1);
		}else if (src_state == 14){
		}else if (src_state == 15){
		}else if (src_state == 16){
		}else if (src_state == 17){
		}else if (src_state == 18){
		}else if (src_state == 19){
		}
	}
#endif

#ifdef	Synthetic_simple_with_neg_100states_kernel
	for(src_state = 0; src_state < 100; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,0,1);
			/***dest 3***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,3,mask,0,0,1);
			/***dest 4***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,4,mask,0,0,1);
			/***dest 5***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,5,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 6***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 8***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 9***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,1,0);
		}else if (src_state == 5){
			/***dest 10***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,1,0);
			/***dest 11***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,11,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 12***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,12,mask,0,1,0);
		}else if (src_state == 7){
			/***dest 13***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,13,mask,0,0,1);
		}else if (src_state == 8){
			/***dest 14***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,14,mask,0,1,0);
		}else if (src_state == 9){
			/***dest 15***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,15,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 16***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,16,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 5***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,5,mask,0,0,1);
			/***dest 17***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,17,mask,0,1,0);
		}else if (src_state == 12){
			/***dest 18***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,18,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 19***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,19,mask,0,0,1);
		}else if (src_state == 14){
			/***dest 20***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,14,20,mask,0,0,1);
		}else if (src_state == 15){
			/***dest 9***/
			mask = match_check(s_char,21,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,9,mask,0,0,1);
			/***dest 21***/
			mask = match_check(s_char,22,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,21,mask,0,1,0);
		}else if (src_state == 16){
			/***dest 22***/
			mask = match_check(s_char,23,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,16,22,mask,0,0,1);
		}else if (src_state == 17){
			/***dest 23***/
			mask = match_check(s_char,24,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,23,mask,0,0,1);
		}else if (src_state == 18){
			/***dest 24***/
			mask = match_check(s_char,25,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,18,24,mask,0,0,1);
		}else if (src_state == 19){
			/***dest 25***/
			mask = match_check(s_char,26,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,25,mask,0,1,0);
		}else if (src_state == 20){
			/***dest 26***/
			mask = match_check(s_char,27,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,26,mask,0,0,1);
		}else if (src_state == 21){
			/***dest 27***/
			mask = match_check(s_char,28,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,21,27,mask,0,1,0);
		}else if (src_state == 22){
			/***dest 28***/
			mask = match_check(s_char,29,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,28,mask,0,0,1);
		}else if (src_state == 23){
			/***dest 29***/
			mask = match_check(s_char,30,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,23,29,mask,0,0,1);
		}else if (src_state == 24){
			/***dest 30***/
			mask = match_check(s_char,31,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,30,mask,0,0,1);
		}else if (src_state == 25){
			/***dest 31***/
			mask = match_check(s_char,32,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,25,31,mask,0,0,1);
		}else if (src_state == 26){
			/***dest 32***/
			mask = match_check(s_char,33,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,32,mask,0,0,1);
		}else if (src_state == 27){
			/***dest 33***/
			mask = match_check(s_char,34,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,27,33,mask,0,0,1);
		}else if (src_state == 28){
			/***dest 34***/
			mask = match_check(s_char,35,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,34,mask,0,0,1);
		}else if (src_state == 29){
			/***dest 35***/
			mask = match_check(s_char,36,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,35,mask,0,1,0);
		}else if (src_state == 30){
			/***dest 36***/
			mask = match_check(s_char,37,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,36,mask,0,1,0);
		}else if (src_state == 31){
			/***dest 37***/
			mask = match_check(s_char,38,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,37,mask,0,0,1);
		}else if (src_state == 32){
			/***dest 38***/
			mask = match_check(s_char,39,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,32,38,mask,0,0,1);
		}else if (src_state == 33){
			/***dest 39***/
			mask = match_check(s_char,40,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,33,39,mask,0,0,1);
		}else if (src_state == 34){
			/***dest 40***/
			mask = match_check(s_char,41,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,34,40,mask,0,0,1);
		}else if (src_state == 35){
			/***dest 41***/
			mask = match_check(s_char,42,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,41,mask,0,0,1);
			/***dest 42***/
			mask = match_check(s_char,43,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,42,mask,0,0,1);
			/***dest 43***/
			mask = match_check(s_char,44,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,43,mask,0,0,1);
			/***dest 44***/
			mask = match_check(s_char,45,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,44,mask,0,0,1);
			/***dest 45***/
			mask = match_check(s_char,46,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,45,mask,0,0,1);
		}else if (src_state == 36){
			/***dest 46***/
			mask = match_check(s_char,47,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,36,46,mask,0,0,1);
		}else if (src_state == 37){
			/***dest 31***/
			mask = match_check(s_char,48,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,31,mask,0,0,1);
			/***dest 47***/
			mask = match_check(s_char,49,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,47,mask,0,1,0);
		}else if (src_state == 38){
			/***dest 48***/
			mask = match_check(s_char,50,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,38,48,mask,0,0,1);
		}else if (src_state == 39){
			/***dest 49***/
			mask = match_check(s_char,51,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,49,mask,0,0,1);
		}else if (src_state == 40){
			/***dest 50***/
			mask = match_check(s_char,52,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,50,mask,0,0,1);
		}else if (src_state == 41){
			/***dest 51***/
			mask = match_check(s_char,53,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,51,mask,0,0,1);
		}else if (src_state == 42){
			/***dest 52***/
			mask = match_check(s_char,54,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,52,mask,0,0,1);
		}else if (src_state == 43){
			/***dest 53***/
			mask = match_check(s_char,55,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,43,53,mask,0,0,1);
		}else if (src_state == 44){
			/***dest 54***/
			mask = match_check(s_char,56,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,54,mask,0,1,0);
		}else if (src_state == 45){
			/***dest 55***/
			mask = match_check(s_char,57,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,55,mask,0,0,1);
		}else if (src_state == 46){
			/***dest 56***/
			mask = match_check(s_char,58,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,56,mask,0,0,1);
		}else if (src_state == 47){
			/***dest 57***/
			mask = match_check(s_char,59,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,47,57,mask,0,0,1);
		}else if (src_state == 48){
			/***dest 58***/
			mask = match_check(s_char,60,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,58,mask,0,1,0);
		}else if (src_state == 49){
			/***dest 59***/
			mask = match_check(s_char,61,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,49,59,mask,0,0,1);
		}else if (src_state == 50){
			/***dest 60***/
			mask = match_check(s_char,62,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,60,mask,0,1,0);
		}else if (src_state == 51){
			/***dest 61***/
			mask = match_check(s_char,63,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,61,mask,0,1,0);
		}else if (src_state == 52){
			/***dest 62***/
			mask = match_check(s_char,64,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,62,mask,0,0,1);
		}else if (src_state == 53){
			/***dest 63***/
			mask = match_check(s_char,65,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,63,mask,0,0,1);
		}else if (src_state == 54){
			/***dest 64***/
			mask = match_check(s_char,66,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,64,mask,0,0,1);
		}else if (src_state == 55){
			/***dest 65***/
			mask = match_check(s_char,67,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,65,mask,0,1,0);
		}else if (src_state == 56){
			/***dest 46***/
			mask = match_check(s_char,68,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,46,mask,0,0,1);
			/***dest 66***/
			mask = match_check(s_char,69,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,66,mask,0,1,0);
		}else if (src_state == 57){
			/***dest 67***/
			mask = match_check(s_char,70,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,67,mask,0,1,0);
		}else if (src_state == 58){
			/***dest 68***/
			mask = match_check(s_char,71,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,68,mask,0,0,1);
		}else if (src_state == 59){
			/***dest 69***/
			mask = match_check(s_char,72,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,59,69,mask,0,0,1);
		}else if (src_state == 60){
			/***dest 70***/
			mask = match_check(s_char,73,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,60,70,mask,0,0,1);
		}else if (src_state == 61){
			/***dest 71***/
			mask = match_check(s_char,74,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,61,71,mask,0,0,1);
		}else if (src_state == 62){
			/***dest 72***/
			mask = match_check(s_char,75,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,62,72,mask,0,0,1);
		}else if (src_state == 63){
			/***dest 73***/
			mask = match_check(s_char,76,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,73,mask,0,1,0);
		}else if (src_state == 64){
			/***dest 74***/
			mask = match_check(s_char,77,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,64,74,mask,0,0,1);
		}else if (src_state == 65){
			/***dest 75***/
			mask = match_check(s_char,78,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,65,75,mask,0,0,1);
		}else if (src_state == 66){
			/***dest 76***/
			mask = match_check(s_char,79,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,66,76,mask,0,0,1);
		}else if (src_state == 67){
			/***dest 77***/
			mask = match_check(s_char,80,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,77,mask,0,0,1);
		}else if (src_state == 68){
			/***dest 78***/
			mask = match_check(s_char,81,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,68,78,mask,0,0,1);
		}else if (src_state == 69){
			/***dest 79***/
			mask = match_check(s_char,82,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,69,79,mask,0,1,0);
		}else if (src_state == 70){
			/***dest 80***/
			mask = match_check(s_char,83,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,80,mask,0,0,1);
		}else if (src_state == 71){
			/***dest 81***/
			mask = match_check(s_char,84,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,71,81,mask,0,1,0);
		}else if (src_state == 72){
			/***dest 82***/
			mask = match_check(s_char,85,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,72,82,mask,0,0,1);
		}else if (src_state == 73){
			/***dest 83***/
			mask = match_check(s_char,86,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,73,83,mask,0,0,1);
		}else if (src_state == 74){
			/***dest 84***/
			mask = match_check(s_char,87,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,74,84,mask,0,0,1);
		}else if (src_state == 75){
			/***dest 85***/
			mask = match_check(s_char,88,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,75,85,mask,0,0,1);
		}else if (src_state == 76){
			/***dest 86***/
			mask = match_check(s_char,89,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,76,86,mask,0,0,1);
		}else if (src_state == 77){
			/***dest 87***/
			mask = match_check(s_char,90,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,77,87,mask,0,0,1);
		}else if (src_state == 78){
			/***dest 88***/
			mask = match_check(s_char,91,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,78,88,mask,0,0,1);
		}else if (src_state == 79){
			/***dest 89***/
			mask = match_check(s_char,92,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,79,89,mask,0,0,1);
		}else if (src_state == 80){
			/***dest 90***/
			mask = match_check(s_char,93,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,80,90,mask,0,1,0);
		}else if (src_state == 81){
			/***dest 91***/
			mask = match_check(s_char,94,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,81,91,mask,0,0,1);
		}else if (src_state == 82){
			/***dest 92***/
			mask = match_check(s_char,95,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,82,92,mask,0,1,0);
		}else if (src_state == 83){
			/***dest 93***/
			mask = match_check(s_char,96,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,83,93,mask,0,0,1);
		}else if (src_state == 84){
			/***dest 94***/
			mask = match_check(s_char,97,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,84,94,mask,0,0,1);
		}else if (src_state == 85){
			/***dest 95***/
			mask = match_check(s_char,98,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,85,95,mask,0,1,0);
		}else if (src_state == 86){
			/***dest 96***/
			mask = match_check(s_char,99,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,86,96,mask,0,0,1);
		}else if (src_state == 87){
			/***dest 97***/
			mask = match_check(s_char,100,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,87,97,mask,0,0,1);
		}else if (src_state == 88){
			/***dest 98***/
			mask = match_check(s_char,101,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,88,98,mask,0,1,0);
		}else if (src_state == 89){
			/***dest 99***/
			mask = match_check(s_char,102,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,89,99,mask,0,0,1);
		}else if (src_state == 90){
		}else if (src_state == 91){
		}else if (src_state == 92){
		}else if (src_state == 93){
		}else if (src_state == 94){
			/***dest 84***/
			mask = match_check(s_char,103,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,94,84,mask,0,0,1);
		}else if (src_state == 95){
		}else if (src_state == 96){
		}else if (src_state == 97){
		}else if (src_state == 98){
		}else if (src_state == 99){
		}
	}
#endif

#ifdef	Synthetic_simple_with_all_20states_kernel
	for(src_state = 0; src_state < 20; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,1,0);
			/***dest 2***/
			character_transitions_update(cfBV, ctof,0,2,mask,1,0,0);
			/***dest 3***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,3,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 1***/
			character_transitions_update(cfBV, ctof,1,1,mask,1,0,0);
			/***dest 4***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,4,mask,0,1,0);
			/***dest 5***/
			character_transitions_update(cfBV, ctof,1,5,mask,1,0,0);
			/***dest 6***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 0***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,0,mask,0,0,1);
			/***dest 8***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 4***/
			character_transitions_update(cfBV, ctof,4,4,mask,1,0,0);
			/***dest 9***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,1,0);
		}else if (src_state == 5){
			/***dest 10***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,0,1);
		}else if (src_state == 6){
			/***dest 11***/
			character_transitions_update(cfBV, ctof,6,11,mask,1,0,0);
		}else if (src_state == 7){
			/***dest 7***/
			character_transitions_update(cfBV, ctof,7,7,mask,1,0,0);
			/***dest 12***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,7,12,mask,0,1,0);
		}else if (src_state == 8){
			/***dest 8***/
			character_transitions_update(cfBV, ctof,8,8,mask,1,0,0);
			/***dest 13***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,13,mask,0,0,1);
			/***dest 14***/
			character_transitions_update(cfBV, ctof,8,14,mask,1,0,0);
			/***dest 15***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,15,mask,0,1,0);
		}else if (src_state == 9){
			/***dest 16***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,16,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 17***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,17,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 18***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,18,mask,0,0,1);
		}else if (src_state == 12){
			/***dest 19***/
			character_transitions_update(cfBV, ctof,12,19,mask,1,0,0);
		}else if (src_state == 13){
			/***dest 8***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,8,mask,0,0,1);
		}else if (src_state == 14){
		}else if (src_state == 15){
		}else if (src_state == 16){
		}else if (src_state == 17){
		}else if (src_state == 18){
		}else if (src_state == 19){
		}
	}
#endif

#ifdef	Synthetic_simple_with_all_100states_kernel
	for(src_state = 0; src_state < 100; src_state++){
 if (src_state == 0){
			/***dest 1***/
			mask = match_check(s_char,0,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,1,mask,0,0,1);
			/***dest 2***/
			mask = match_check(s_char,1,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,2,mask,0,0,1);
			/***dest 3***/
			character_transitions_update(cfBV, ctof,0,3,mask,1,0,0);
			/***dest 4***/
			mask = match_check(s_char,2,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,4,mask,0,0,1);
			/***dest 5***/
			mask = match_check(s_char,3,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,0,5,mask,0,0,1);
		}else if (src_state == 1){
			/***dest 1***/
			character_transitions_update(cfBV, ctof,1,1,mask,1,0,0);
			/***dest 6***/
			mask = match_check(s_char,4,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,1,6,mask,0,0,1);
		}else if (src_state == 2){
			/***dest 7***/
			mask = match_check(s_char,5,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,2,7,mask,0,0,1);
		}else if (src_state == 3){
			/***dest 8***/
			mask = match_check(s_char,6,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,3,8,mask,0,0,1);
		}else if (src_state == 4){
			/***dest 4***/
			character_transitions_update(cfBV, ctof,4,4,mask,1,0,0);
			/***dest 9***/
			mask = match_check(s_char,7,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,4,9,mask,0,1,0);
		}else if (src_state == 5){
			/***dest 5***/
			character_transitions_update(cfBV, ctof,5,5,mask,1,0,0);
			/***dest 10***/
			mask = match_check(s_char,8,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,5,10,mask,0,1,0);
			/***dest 11***/
			character_transitions_update(cfBV, ctof,5,11,mask,1,0,0);
		}else if (src_state == 6){
			/***dest 12***/
			mask = match_check(s_char,9,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,6,12,mask,0,1,0);
		}else if (src_state == 7){
			/***dest 7***/
			character_transitions_update(cfBV, ctof,7,7,mask,1,0,0);
			/***dest 13***/
			character_transitions_update(cfBV, ctof,7,13,mask,1,0,0);
		}else if (src_state == 8){
			/***dest 14***/
			mask = match_check(s_char,10,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,8,14,mask,0,1,0);
		}else if (src_state == 9){
			/***dest 15***/
			mask = match_check(s_char,11,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,9,15,mask,0,0,1);
		}else if (src_state == 10){
			/***dest 16***/
			mask = match_check(s_char,12,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,10,16,mask,0,0,1);
		}else if (src_state == 11){
			/***dest 5***/
			mask = match_check(s_char,13,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,5,mask,0,0,1);
			/***dest 17***/
			mask = match_check(s_char,14,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,11,17,mask,0,1,0);
		}else if (src_state == 12){
			/***dest 18***/
			mask = match_check(s_char,15,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,12,18,mask,0,0,1);
		}else if (src_state == 13){
			/***dest 19***/
			mask = match_check(s_char,16,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,13,19,mask,0,0,1);
		}else if (src_state == 14){
			/***dest 14***/
			character_transitions_update(cfBV, ctof,14,14,mask,1,0,0);
			/***dest 20***/
			character_transitions_update(cfBV, ctof,14,20,mask,1,0,0);
		}else if (src_state == 15){
			/***dest 9***/
			mask = match_check(s_char,17,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,9,mask,0,0,1);
			/***dest 21***/
			mask = match_check(s_char,18,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,15,21,mask,0,1,0);
		}else if (src_state == 16){
			/***dest 22***/
			character_transitions_update(cfBV, ctof,16,22,mask,1,0,0);
		}else if (src_state == 17){
			/***dest 23***/
			mask = match_check(s_char,19,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,17,23,mask,0,0,1);
		}else if (src_state == 18){
			/***dest 18***/
			character_transitions_update(cfBV, ctof,18,18,mask,1,0,0);
			/***dest 24***/
			character_transitions_update(cfBV, ctof,18,24,mask,1,0,0);
		}else if (src_state == 19){
			/***dest 25***/
			mask = match_check(s_char,20,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,19,25,mask,0,1,0);
		}else if (src_state == 20){
			/***dest 26***/
			mask = match_check(s_char,21,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,20,26,mask,0,0,1);
		}else if (src_state == 21){
			/***dest 21***/
			character_transitions_update(cfBV, ctof,21,21,mask,1,0,0);
			/***dest 27***/
			mask = match_check(s_char,22,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,21,27,mask,0,1,0);
		}else if (src_state == 22){
			/***dest 22***/
			character_transitions_update(cfBV, ctof,22,22,mask,1,0,0);
			/***dest 28***/
			mask = match_check(s_char,23,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,22,28,mask,0,0,1);
		}else if (src_state == 23){
			/***dest 29***/
			character_transitions_update(cfBV, ctof,23,29,mask,1,0,0);
		}else if (src_state == 24){
			/***dest 30***/
			mask = match_check(s_char,24,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,24,30,mask,0,0,1);
		}else if (src_state == 25){
			/***dest 25***/
			character_transitions_update(cfBV, ctof,25,25,mask,1,0,0);
			/***dest 31***/
			character_transitions_update(cfBV, ctof,25,31,mask,1,0,0);
		}else if (src_state == 26){
			/***dest 32***/
			mask = match_check(s_char,25,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,26,32,mask,0,0,1);
		}else if (src_state == 27){
			/***dest 33***/
			character_transitions_update(cfBV, ctof,27,33,mask,1,0,0);
		}else if (src_state == 28){
			/***dest 34***/
			mask = match_check(s_char,26,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,28,34,mask,0,0,1);
		}else if (src_state == 29){
			/***dest 35***/
			mask = match_check(s_char,27,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,29,35,mask,0,1,0);
		}else if (src_state == 30){
			/***dest 36***/
			mask = match_check(s_char,28,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,30,36,mask,0,1,0);
		}else if (src_state == 31){
			/***dest 37***/
			mask = match_check(s_char,29,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,31,37,mask,0,0,1);
		}else if (src_state == 32){
			/***dest 32***/
			character_transitions_update(cfBV, ctof,32,32,mask,1,0,0);
			/***dest 38***/
			character_transitions_update(cfBV, ctof,32,38,mask,1,0,0);
		}else if (src_state == 33){
			/***dest 39***/
			mask = match_check(s_char,30,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,33,39,mask,0,0,1);
		}else if (src_state == 34){
			/***dest 40***/
			character_transitions_update(cfBV, ctof,34,40,mask,1,0,0);
		}else if (src_state == 35){
			/***dest 35***/
			character_transitions_update(cfBV, ctof,35,35,mask,1,0,0);
			/***dest 41***/
			character_transitions_update(cfBV, ctof,35,41,mask,1,0,0);
			/***dest 42***/
			character_transitions_update(cfBV, ctof,35,42,mask,1,0,0);
			/***dest 43***/
			mask = match_check(s_char,31,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,43,mask,0,0,1);
			/***dest 44***/
			mask = match_check(s_char,32,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,44,mask,0,0,1);
			/***dest 45***/
			mask = match_check(s_char,33,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,35,45,mask,0,0,1);
		}else if (src_state == 36){
			/***dest 46***/
			character_transitions_update(cfBV, ctof,36,46,mask,1,0,0);
		}else if (src_state == 37){
			/***dest 31***/
			mask = match_check(s_char,34,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,31,mask,0,0,1);
			/***dest 47***/
			mask = match_check(s_char,35,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,37,47,mask,0,1,0);
		}else if (src_state == 38){
			/***dest 48***/
			mask = match_check(s_char,36,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,38,48,mask,0,0,1);
		}else if (src_state == 39){
			/***dest 39***/
			character_transitions_update(cfBV, ctof,39,39,mask,1,0,0);
			/***dest 49***/
			mask = match_check(s_char,37,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,39,49,mask,0,0,1);
		}else if (src_state == 40){
			/***dest 50***/
			mask = match_check(s_char,38,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,40,50,mask,0,0,1);
		}else if (src_state == 41){
			/***dest 41***/
			character_transitions_update(cfBV, ctof,41,41,mask,1,0,0);
			/***dest 51***/
			mask = match_check(s_char,39,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,41,51,mask,0,0,1);
		}else if (src_state == 42){
			/***dest 52***/
			mask = match_check(s_char,40,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,42,52,mask,0,0,1);
		}else if (src_state == 43){
			/***dest 53***/
			character_transitions_update(cfBV, ctof,43,53,mask,1,0,0);
		}else if (src_state == 44){
			/***dest 54***/
			mask = match_check(s_char,41,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,44,54,mask,0,1,0);
		}else if (src_state == 45){
			/***dest 55***/
			mask = match_check(s_char,42,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,45,55,mask,0,0,1);
		}else if (src_state == 46){
			/***dest 46***/
			character_transitions_update(cfBV, ctof,46,46,mask,1,0,0);
			/***dest 56***/
			mask = match_check(s_char,43,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,46,56,mask,0,0,1);
		}else if (src_state == 47){
			/***dest 57***/
			character_transitions_update(cfBV, ctof,47,57,mask,1,0,0);
		}else if (src_state == 48){
			/***dest 58***/
			mask = match_check(s_char,44,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,48,58,mask,0,1,0);
		}else if (src_state == 49){
			/***dest 59***/
			character_transitions_update(cfBV, ctof,49,59,mask,1,0,0);
		}else if (src_state == 50){
			/***dest 50***/
			character_transitions_update(cfBV, ctof,50,50,mask,1,0,0);
			/***dest 60***/
			mask = match_check(s_char,45,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,50,60,mask,0,1,0);
		}else if (src_state == 51){
			/***dest 61***/
			mask = match_check(s_char,46,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,51,61,mask,0,1,0);
		}else if (src_state == 52){
			/***dest 52***/
			character_transitions_update(cfBV, ctof,52,52,mask,1,0,0);
			/***dest 62***/
			mask = match_check(s_char,47,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,52,62,mask,0,0,1);
		}else if (src_state == 53){
			/***dest 63***/
			mask = match_check(s_char,48,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,53,63,mask,0,0,1);
		}else if (src_state == 54){
			/***dest 64***/
			mask = match_check(s_char,49,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,54,64,mask,0,0,1);
		}else if (src_state == 55){
			/***dest 55***/
			character_transitions_update(cfBV, ctof,55,55,mask,1,0,0);
			/***dest 65***/
			mask = match_check(s_char,50,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,55,65,mask,0,1,0);
		}else if (src_state == 56){
			/***dest 46***/
			mask = match_check(s_char,51,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,46,mask,0,0,1);
			/***dest 66***/
			mask = match_check(s_char,52,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,56,66,mask,0,1,0);
		}else if (src_state == 57){
			/***dest 67***/
			mask = match_check(s_char,53,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,57,67,mask,0,1,0);
		}else if (src_state == 58){
			/***dest 58***/
			character_transitions_update(cfBV, ctof,58,58,mask,1,0,0);
			/***dest 68***/
			mask = match_check(s_char,54,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,58,68,mask,0,0,1);
		}else if (src_state == 59){
			/***dest 69***/
			mask = match_check(s_char,55,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,59,69,mask,0,0,1);
		}else if (src_state == 60){
			/***dest 70***/
			character_transitions_update(cfBV, ctof,60,70,mask,1,0,0);
		}else if (src_state == 61){
			/***dest 71***/
			mask = match_check(s_char,56,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,61,71,mask,0,0,1);
		}else if (src_state == 62){
			/***dest 72***/
			character_transitions_update(cfBV, ctof,62,72,mask,1,0,0);
		}else if (src_state == 63){
			/***dest 73***/
			mask = match_check(s_char,57,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,63,73,mask,0,1,0);
		}else if (src_state == 64){
			/***dest 64***/
			character_transitions_update(cfBV, ctof,64,64,mask,1,0,0);
			/***dest 74***/
			character_transitions_update(cfBV, ctof,64,74,mask,1,0,0);
		}else if (src_state == 65){
			/***dest 75***/
			character_transitions_update(cfBV, ctof,65,75,mask,1,0,0);
		}else if (src_state == 66){
			/***dest 76***/
			character_transitions_update(cfBV, ctof,66,76,mask,1,0,0);
		}else if (src_state == 67){
			/***dest 67***/
			character_transitions_update(cfBV, ctof,67,67,mask,1,0,0);
			/***dest 77***/
			mask = match_check(s_char,58,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,67,77,mask,0,0,1);
		}else if (src_state == 68){
			/***dest 78***/
			character_transitions_update(cfBV, ctof,68,78,mask,1,0,0);
		}else if (src_state == 69){
			/***dest 79***/
			mask = match_check(s_char,59,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,69,79,mask,0,1,0);
		}else if (src_state == 70){
			/***dest 70***/
			character_transitions_update(cfBV, ctof,70,70,mask,1,0,0);
			/***dest 80***/
			mask = match_check(s_char,60,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,70,80,mask,0,0,1);
		}else if (src_state == 71){
			/***dest 71***/
			character_transitions_update(cfBV, ctof,71,71,mask,1,0,0);
			/***dest 81***/
			mask = match_check(s_char,61,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,71,81,mask,0,1,0);
		}else if (src_state == 72){
			/***dest 82***/
			mask = match_check(s_char,62,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,72,82,mask,0,0,1);
		}else if (src_state == 73){
			/***dest 73***/
			character_transitions_update(cfBV, ctof,73,73,mask,1,0,0);
			/***dest 83***/
			mask = match_check(s_char,63,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,73,83,mask,0,0,1);
		}else if (src_state == 74){
			/***dest 84***/
			mask = match_check(s_char,64,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,74,84,mask,0,0,1);
		}else if (src_state == 75){
			/***dest 75***/
			character_transitions_update(cfBV, ctof,75,75,mask,1,0,0);
			/***dest 85***/
			mask = match_check(s_char,65,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,75,85,mask,0,0,1);
		}else if (src_state == 76){
			/***dest 86***/
			mask = match_check(s_char,66,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,76,86,mask,0,0,1);
		}else if (src_state == 77){
			/***dest 87***/
			character_transitions_update(cfBV, ctof,77,87,mask,1,0,0);
		}else if (src_state == 78){
			/***dest 88***/
			mask = match_check(s_char,67,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,78,88,mask,0,0,1);
		}else if (src_state == 79){
			/***dest 89***/
			character_transitions_update(cfBV, ctof,79,89,mask,1,0,0);
		}else if (src_state == 80){
			/***dest 90***/
			mask = match_check(s_char,68,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,80,90,mask,0,1,0);
		}else if (src_state == 81){
			/***dest 91***/
			mask = match_check(s_char,69,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,81,91,mask,0,0,1);
		}else if (src_state == 82){
			/***dest 92***/
			mask = match_check(s_char,70,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,82,92,mask,0,1,0);
		}else if (src_state == 83){
			/***dest 93***/
			mask = match_check(s_char,71,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,83,93,mask,0,0,1);
		}else if (src_state == 84){
			/***dest 94***/
			mask = match_check(s_char,72,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,84,94,mask,0,0,1);
		}else if (src_state == 85){
			/***dest 95***/
			mask = match_check(s_char,73,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,85,95,mask,0,1,0);
		}else if (src_state == 86){
			/***dest 96***/
			mask = match_check(s_char,74,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,86,96,mask,0,0,1);
		}else if (src_state == 87){
			/***dest 97***/
			mask = match_check(s_char,75,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,87,97,mask,0,0,1);
		}else if (src_state == 88){
			/***dest 98***/
			mask = match_check(s_char,76,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,88,98,mask,0,1,0);
		}else if (src_state == 89){
			/***dest 99***/
			mask = match_check(s_char,77,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,89,99,mask,0,0,1);
		}else if (src_state == 90){
		}else if (src_state == 91){
		}else if (src_state == 92){
		}else if (src_state == 93){
		}else if (src_state == 94){
			/***dest 84***/
			mask = match_check(s_char,78,1,preprocessed_input,ref_block_count,batch_count);
			character_transitions_update(cfBV, ctof,94,84,mask,0,0,1);
		}else if (src_state == 95){
		}else if (src_state == 96){
		}else if (src_state == 97){
		}else if (src_state == 98){
		}else if (src_state == 99){
		}
	}
#endif

	update_StateVector(cfBV, ctof, bit_chunks_per_state_vector, accepting_states_count);
}
