#ifndef GENERAL_NFA_FUNCTIONS_CUH
#define GENERAL_NFA_FUNCTIONS_CUH

#include <string.h>
#include "general_stdinc.h"
///#include "general_kernels.h"


struct preprocessed_full_reference_char_sequence{
	unsigned SOA_chunks_7_word[MAX_SOA_CHUNCK_COUNT];
	unsigned SOA_chunks_6_word[MAX_SOA_CHUNCK_COUNT];
	unsigned SOA_chunks_5_word[MAX_SOA_CHUNCK_COUNT];
	unsigned SOA_chunks_4_word[MAX_SOA_CHUNCK_COUNT];
	unsigned SOA_chunks_3_word[MAX_SOA_CHUNCK_COUNT];
	unsigned SOA_chunks_2_word[MAX_SOA_CHUNCK_COUNT];
	unsigned SOA_chunks_1_word[MAX_SOA_CHUNCK_COUNT];
	unsigned SOA_chunks_0_word[MAX_SOA_CHUNCK_COUNT];
};

int general_initialize();

int general_preprocessed_to_device();

int general_inputstreaming();


#endif //GENERAL_NFA_FUNCTIONS_CUH
