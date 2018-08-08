#ifndef TOPOLOGY_FUNCTION_CUH_
#define TOPOLOGY_FUNCTION_CUH_

#include "general_stdinc.h"
#include "general_kernels.h"

#include "cudaCheckError.h"

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
					unsigned batch_count);

#endif /*TOPOLOGY_FUNCTION_CUH_*/
