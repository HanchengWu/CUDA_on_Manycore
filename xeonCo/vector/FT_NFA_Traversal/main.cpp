#include "main.h"


struct general_config cf0; //M: should move from here!
int stream_count_from_command_line = 0;


/*function declaration*/
char * load_streaming_sequence_dividestream(char * input_sequence_filename);
char * load_streaming_sequence_multitrace();
int general_result_vector_from_device();
int general_postprocess_and_validate();
void* read_compiler_file();
static int parse_arguments(int argc, char **argv);

int general_initialize(){

	/*reported variable initiation*/
	cf0.start=0;
	cf0.stop=0;
	cf0.preprocessing=0;
	cf0.stream_to_dev=0;
	cf0.kernel=0;
	cf0.result_from_dev=0;
	cf0.post_processing=0;
	char * out_name = strdup("final_test_general.csv");
	cf0.final_test_outfile = fopen(out_name, "a");	
	if(!(cf0.final_test_outfile)){
		printf("Final test output file not opening.  Exiting!\n");
		exit(-1);
	}
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	sprintf(cf0.start_stamp, "%d-%d-%d %d:%d:%d\t", tm.tm_mon + 1, tm.tm_mday, tm.tm_year + 1900, tm.tm_hour, tm.tm_min, tm.tm_sec);
	cf0.start=gettime();//start preprocessing clock


	cf0.char_filled_ints_per_packet = cf0.packet_size / sizeof(unsigned);



	/*compiler's file inputs initiation*/
	cf0.nfa_size=0;
	cf0.ref_block_count=0;
	cf0.batch_count =0;
	cf0.batch_size=0;		//(not used)i have not replaced sizeof(unsigned) that is used instead of batch_size in the other function. In case of use of 16 bits for thread, make this change
	cf0.threads_per_block=0; //SIMDwidth
	cf0.warp_efficient_stream_count=0;
	cf0.occupancy_efficient_stream_count=0;
	cf0.accepting_states_count=0;
	cf0.SOA_chunk_count=0;

	/*compiler's file inputs reading from file*/
	/*read preprocessed data from compiler's input file*/
	cf0.fc_preprocessed_input_h = (struct preprocessed_full_reference_char_sequence *)read_compiler_file();

	/*if occupancy_efficient_stream_count is mentioned in command line it will be read from command line*/
	if (stream_count_from_command_line > 0) cf0.occupancy_efficient_stream_count =stream_count_from_command_line;

	cf0.streams_count = cf0.warp_efficient_stream_count*cf0.occupancy_efficient_stream_count;
	cf0.blocks_count = cf0.occupancy_efficient_stream_count * cf0.ref_block_count;
	cf0.bit_chunks_per_state_vector = (cf0.threads_per_block/*cf0.batch_size*sizeof(unsigned)*/*cf0.nfa_size) ;

	/*read streaming sequences characters from input file*/
#ifdef DIVIDE_TRACE
	cf0.fc_streaming_sequences_h = (char *)load_streaming_sequence_dividestream(cf0.stream_sequence_filename[0]);
#else
	cf0.fc_streaming_sequences_h = (char *)load_streaming_sequence_multitrace();
#endif
	fprintf(stderr,"\n---Initiation results---\n");

	fprintf(stderr,"Compiler's output filename = %s\n", cf0.compiler_output_filename);
	//fprintf(stderr,"Stream sequence filename = \n");
	//for(int i=0; i< cf0.trace_num; i++) printf("					%s\n",cf0.stream_sequence_filename[i]);
	fprintf(stderr, "packet_per_kernel_launch = %d\n",cf0.packets_per_kernel_launch);
	fprintf(stderr, "packet_size = %d\n",cf0.packet_size);
	fprintf(stderr,"char_filled_ints_per_packet = %d\n", cf0.char_filled_ints_per_packet);
	fprintf(stderr,"gpu_device = %d\n", cf0.gpu_device);
	fprintf(stderr, "Application : %s \n",cf0.application);
	fprintf(stderr,"\nnfa_size = %d\n", cf0.nfa_size);
	fprintf(stderr,"ref_block_count = %d\n", cf0.ref_block_count);
	fprintf(stderr,"batch_count = %d\n", cf0.batch_count);
	fprintf(stderr,"batch_size = %d (Bytes)\n", cf0.batch_size);
	fprintf(stderr,"threads_per_block = %d\n", cf0.threads_per_block);
	fprintf(stderr,"warp_efficient_stream_count = %d\n", cf0.warp_efficient_stream_count);
	fprintf(stderr,"occupancy_efficient_stream_count = %d\n", cf0.occupancy_efficient_stream_count);
	fprintf(stderr,"accepting_states_count = %d\n", cf0.accepting_states_count);
	fprintf(stderr,"SOA_chunk_count = %d\n", cf0.SOA_chunk_count); //use to read preprocessed sequence from file
	fprintf(stderr,"streams_count = %d\n", cf0.streams_count);
	fprintf(stderr,"blocks_count = %d\n", cf0.blocks_count);
	fprintf(stderr,"bit_chunks_per_state_vector = %d\n", cf0.bit_chunks_per_state_vector);

	fprintf(stderr,"cf0.k = %d\n", cf0.k);
	fprintf(stderr,"cf0.d = %d\n", cf0.d);

	//printf("\nPress any key to continue...\n");

	/*delete the stream_sequence_filename*/
	//for(int i=0; i< cf0.trace_num; i++) delete [] cf0.stream_sequence_filename[i];
	delete [] cf0.stream_sequence_filename;

//****************************************************************************************//
	return 0;
}

void* read_compiler_file(){

	FILE * input_file;
	input_file = fopen(cf0.compiler_output_filename, "r");
	if(!input_file){
		printf("\nthe input file from compiler was not successfully opened!\n");
	}
	fscanf(input_file,"%s",cf0.application);
	fscanf(input_file,"%d",&(cf0.nfa_size));
	fscanf(input_file,"%d",&(cf0.ref_block_count));
	fscanf(input_file,"%d",&(cf0.batch_count));
	fscanf(input_file,"%d",&(cf0.batch_size));
	fscanf(input_file,"%d",&(cf0.threads_per_block));
	fscanf(input_file,"%d",&(cf0.warp_efficient_stream_count));
	fscanf(input_file,"%d",&(cf0.occupancy_efficient_stream_count));
	fscanf(input_file,"%d",&(cf0.accepting_states_count));
	fscanf(input_file,"%d",&(cf0.SOA_chunk_count));

	struct preprocessed_full_reference_char_sequence * NFA_full_char_sequences = NULL;
	NFA_full_char_sequences = (struct preprocessed_full_reference_char_sequence *)calloc(cf0.ref_block_count, sizeof(struct preprocessed_full_reference_char_sequence));

	for (int i=0; i < cf0.ref_block_count; i++){
		/*array allocation and initiation*/
		char * array_7 = (char *)calloc( 4 * cf0.SOA_chunk_count +1, sizeof(char) );	//+1 since fscanf might attach an end of sequence char to arrays
		char * array_6 = (char *)calloc( 4 * cf0.SOA_chunk_count +1, sizeof(char) );
		char * array_5 = (char *)calloc( 4 * cf0.SOA_chunk_count +1, sizeof(char) );
		char * array_4 = (char *)calloc( 4 * cf0.SOA_chunk_count +1, sizeof(char) );
		char * array_3 = (char *)calloc( 4 * cf0.SOA_chunk_count +1, sizeof(char) );
		char * array_2 = (char *)calloc( 4 * cf0.SOA_chunk_count +1, sizeof(char) );
		char * array_1 = (char *)calloc( 4 * cf0.SOA_chunk_count +1, sizeof(char) );
		char * array_0 = (char *)calloc( 4 * cf0.SOA_chunk_count +1, sizeof(char) );
		/*filling char arrays*/
		char c = fgetc(input_file);//this should be \n
		for(int j=0; j < 4*cf0.SOA_chunk_count ; j++) array_7[j] = fgetc(input_file);
		c = fgetc(input_file);//this should be \n
		for(int j=0; j < 4*cf0.SOA_chunk_count ; j++) array_6[j] = fgetc(input_file);
		c = fgetc(input_file);//this should be \n
		for(int j=0; j < 4*cf0.SOA_chunk_count ; j++) array_5[j] = fgetc(input_file);
		c = fgetc(input_file);//this should be \n
		for(int j=0; j < 4*cf0.SOA_chunk_count ; j++) array_4[j] = fgetc(input_file);
		c = fgetc(input_file);//this should be \n
		for(int j=0; j < 4*cf0.SOA_chunk_count ; j++) array_3[j] = fgetc(input_file);
		c = fgetc(input_file);//this should be \n
		for(int j=0; j < 4*cf0.SOA_chunk_count ; j++) array_2[j] = fgetc(input_file);
		c = fgetc(input_file); //this should be \n
		for(int j=0; j < 4*cf0.SOA_chunk_count ; j++) array_1[j] = fgetc(input_file);
		c = fgetc(input_file);//this should be \n
		for(int j=0; j < 4*cf0.SOA_chunk_count ; j++) array_0[j] = fgetc(input_file);
		/*move to the preprocessed struct*/
		memcpy( (void *)  ( NFA_full_char_sequences[i].SOA_chunks_7_word ), (void *)array_7 , 4*cf0.SOA_chunk_count*sizeof(char)  );
		memcpy( (void *)  ( NFA_full_char_sequences[i].SOA_chunks_6_word ), (void *)array_6 , 4*cf0.SOA_chunk_count*sizeof(char)  );
		memcpy( (void *)  ( NFA_full_char_sequences[i].SOA_chunks_5_word ), (void *)array_5 , 4*cf0.SOA_chunk_count*sizeof(char)  );
		memcpy( (void *)  ( NFA_full_char_sequences[i].SOA_chunks_4_word ), (void *)array_4 , 4*cf0.SOA_chunk_count*sizeof(char)  );
		memcpy( (void *)  ( NFA_full_char_sequences[i].SOA_chunks_3_word ), (void *)array_3 , 4*cf0.SOA_chunk_count*sizeof(char)  );
		memcpy( (void *)  ( NFA_full_char_sequences[i].SOA_chunks_2_word ), (void *)array_2 , 4*cf0.SOA_chunk_count*sizeof(char)  );
		memcpy( (void *)  ( NFA_full_char_sequences[i].SOA_chunks_1_word ), (void *)array_1 , 4*cf0.SOA_chunk_count*sizeof(char)  );
		memcpy( (void *)  ( NFA_full_char_sequences[i].SOA_chunks_0_word ), (void *)array_0 , 4*cf0.SOA_chunk_count*sizeof(char)  );
		/*free the char arrays*/
		free(array_7);
		free(array_6);
		free(array_5);
		free(array_4);
		free(array_3);
		free(array_2);
		free(array_1);
		free(array_0);
	}
	fclose(input_file);
	return (void *)NFA_full_char_sequences;
}
/* reads streams into a single array from input file*/
char * load_streaming_sequence_dividestream(char * input_sequence_filename){

	char * stream_sequences = (char *)calloc(cf0.packets_per_kernel_launch * cf0.packet_size * cf0.streams_count , sizeof(char));
	FILE * infile1;
	infile1 = fopen(input_sequence_filename, "r");
	if(!infile1){
		printf("Error opening streaming sequence input file.  Check if it exists!  Exiting!\n");
		exit(-1);
	}
	unsigned result = fread (stream_sequences,1,cf0.packet_size * cf0.packets_per_kernel_launch * cf0.streams_count ,infile1);
	 if (result != cf0.packet_size*cf0.packets_per_kernel_launch* cf0.streams_count) fputs ("Not enough inputs for the streams\n",stderr);
	fclose(infile1);
	return stream_sequences;
}

char * load_streaming_sequence_multitrace(){

	FILE ** trace_file =(FILE **) malloc(cf0.trace_num*sizeof(FILE *));
	char * stream_sequences = (char *)calloc(cf0.packets_per_kernel_launch * cf0.packet_size * cf0.streams_count , sizeof(char));
	for(int i=0;i <cf0.trace_num;i++) trace_file[i] = fopen(cf0.stream_sequence_filename[i], "r");

	unsigned origin = ftell(trace_file[0]);

	for (unsigned s=0;s<cf0.streams_count;s++){

			unsigned t = s % cf0.trace_num;

			char * buffer;
			buffer = (char *) malloc (sizeof(char)*cf0.packets_per_kernel_launch * cf0.packet_size);
			fseek(trace_file[t], origin, SEEK_SET);//reposition position indicator

			unsigned result =fread(buffer, 1, cf0.packets_per_kernel_launch * cf0.packet_size ,trace_file[t]);

			if (result != cf0.packets_per_kernel_launch * cf0.packet_size) fputs ("Not enough inputs for the streams in trace file\n",stderr);
			for(int j=0;j<cf0.packets_per_kernel_launch * cf0.packet_size;j++) stream_sequences[s*(cf0.packets_per_kernel_launch*cf0.packet_size)+j]=buffer[j];

			free(buffer);
	}

	for(int i=0;i <cf0.trace_num;i++) fclose(trace_file[i]);
	delete trace_file;

	return stream_sequences;
}

int general_preprocessed_to_device(){

	///cudaCheckError( cudaMalloc(&(cf0.fc_preprocessed_input_d), sizeof(struct preprocessed_full_reference_char_sequence)*cf0.ref_block_count ) , __LINE__, __FILE__);
	///cudaCheckError( cudaMemcpy(cf0.fc_preprocessed_input_d, cf0.fc_preprocessed_input_h, sizeof(struct preprocessed_full_reference_char_sequence)*cf0.ref_block_count , cudaMemcpyHostToDevice), __LINE__, __FILE__);

	cf0.stop=gettime();
	cf0.preprocessing=cf0.stop-cf0.start;

	return 0;
}

int general_stream_burst_to_device(){
	//Memory operations on/to device
	cf0.start=gettime();
	/*result bit vector allocation and initiation*/
#ifdef STATE_VECTOR_DEBUG
	cf0.result_bit_vector_h = (unsigned *)calloc(          cf0.bit_chunks_per_state_vector , sizeof(unsigned) );
	///cudaCheckError( cudaMalloc(&(cf0.result_bit_vector_d), cf0.bit_chunks_per_state_vector * sizeof(unsigned) ) , __LINE__, __FILE__);
	///cudaCheckError( cudaMemset(cf0.result_bit_vector_d, 0, cf0.bit_chunks_per_state_vector * sizeof(unsigned) ) , __LINE__, __FILE__);
#else
	cf0.result_bit_vector_h = (unsigned *)calloc(          (cf0.accepting_states_count)*cf0.blocks_count*cf0.threads_per_block*cf0.packets_per_kernel_launch,sizeof(unsigned) );
	///cudaCheckError( cudaMalloc(&(cf0.result_bit_vector_d), (cf0.accepting_states_count)*cf0.blocks_count*cf0.threads_per_block*cf0.packets_per_kernel_launch*sizeof(unsigned) ) , __LINE__, __FILE__);
	///cudaCheckError( cudaMemset(cf0.result_bit_vector_d, 0, (cf0.accepting_states_count)*cf0.blocks_count*cf0.threads_per_block*cf0.packets_per_kernel_launch*sizeof(unsigned) ) , __LINE__, __FILE__);
#endif

	///cudaCheckError( cudaMalloc(&(cf0.fc_streaming_sequences_d), sizeof(char)*cf0.packet_size*cf0.packets_per_kernel_launch *cf0.streams_count) , __LINE__, __FILE__);
	///cudaCheckError( cudaMemcpy(cf0.fc_streaming_sequences_d, cf0.fc_streaming_sequences_h, sizeof(char)*cf0.packet_size*cf0.packets_per_kernel_launch*cf0.streams_count , cudaMemcpyHostToDevice), __LINE__, __FILE__);
	cf0.stop=gettime();
	cf0.stream_to_dev=cf0.stop-cf0.start;

	return 0;
}

int general_nfa_execute(){

	cf0.start = gettime();


	/*fixed_topology_kernel<<<cf0.blocks_count,cf0.threads_per_block,2*sizeof(unsigned)*cf0.bit_chunks_per_state_vector>>>(//2 multiplier because of requiring both current and future state vectors
																									   cf0.result_bit_vector_d,
																									   cf0.fc_streaming_sequences_d,
																									   cf0.fc_preprocessed_input_d,
																									   cf0.bit_chunks_per_state_vector,
																									   cf0.char_filled_ints_per_packet,
																									   cf0.packets_per_kernel_launch,
																									   cf0.warp_efficient_stream_count,
																									   cf0.occupancy_efficient_stream_count,
																									   cf0.ref_block_count,
																									   cf0.batch_count,
																									   cf0.accepting_states_count  );

	cudaDeviceSynchronize();*/
	cf0.stop=gettime();
	cf0.kernel+=cf0.stop-cf0.start;

	cf0.start = gettime();

	general_result_vector_from_device();//Get the results!! Note this call contains a cudaDeviceSynchronize() after the mem operations!  Also, this includes frees!

	cf0.stop = gettime();
	cf0.result_from_dev+=cf0.stop-cf0.start;

	///cudaProfilerStop();//No profiling of postprocess stage

	cf0.start=gettime();

	general_postprocess_and_validate();

	cf0.stop=gettime();
	cf0.post_processing+=cf0.stop-cf0.start;

	fprintf(cf0.final_test_outfile, "%s	", cf0.start_stamp);
	fprintf(cf0.final_test_outfile, "%s	", cf0.application);
	fprintf(cf0.final_test_outfile, "blocks=%d\t", cf0.blocks_count);
	fprintf(cf0.final_test_outfile, "%3.6f\t", cf0.preprocessing);
	fprintf(cf0.final_test_outfile, "%3.6f\t", cf0.stream_to_dev);
	fprintf(cf0.final_test_outfile, "%3.6f\t", cf0.kernel);
	fprintf(cf0.final_test_outfile, "%3.6f\t", cf0.result_from_dev);
	fprintf(cf0.final_test_outfile, "%3.6f\t\n", cf0.post_processing);


	///cudaProfilerStart();

	return 0;
}

int general_inputstreaming(){

	/*no multiple kernel calls for now*/
	
	/*move the result bit vector and the input sequence to the device*/
	general_stream_burst_to_device();

	/*launch kernel and synchronize threads and move the result to host and post process and pront the statistical info to file */
	general_nfa_execute();

	/*moved this here (from general_result_vector_from_device() since in infant thsi is not computed as part of config.result_from_dev timing)*/
	free(cf0.fc_preprocessed_input_h);
	///cudaCheckError( cudaFree(cf0.fc_preprocessed_input_d) , __LINE__, __FILE__);

	///cudaProfilerStop();


	return 0;
}


int general_result_vector_from_device(){
	//Memory operations from device
	#ifdef STATE_VECTOR_DEBUG
		///cudaCheckError( cudaMemcpy(cf0.result_bit_vector_h, cf0.result_bit_vector_d, cf0.bit_chunks_per_state_vector * sizeof(unsigned) , cudaMemcpyDeviceToHost), __LINE__, __FILE__);
	#else
		///cudaCheckError( cudaMemcpy(cf0.result_bit_vector_h, cf0.result_bit_vector_d, (cf0.accepting_states_count)*cf0.blocks_count*cf0.threads_per_block*cf0.packets_per_kernel_launch*sizeof(unsigned), cudaMemcpyDeviceToHost), __LINE__, __FILE__);
	#endif
	///cudaDeviceSynchronize();

	///cudaFree(cf0.result_bit_vector_d);

	free(cf0.fc_streaming_sequences_h);
	///cudaCheckError( cudaFree(cf0.fc_streaming_sequences_d) , __LINE__, __FILE__);


	return 0;
}

unsigned numberOfSetBits(unsigned i){
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

int general_postprocess_and_validate(){

	unsigned total_matches=0;
	unsigned line_matches=0;
#ifdef POSTPROC


	/*print the accepting portion of status vector*/
	if(!(strcmp(cf0.application,"Levenshtein")== 0) && !(strcmp(cf0.application,"Hamming")== 0)){

#ifndef STATE_VECTOR_DEBUG

		for(int i=0; i < (cf0.accepting_states_count)*cf0.threads_per_block*cf0.blocks_count*cf0.packets_per_kernel_launch; i++){
			line_matches = numberOfSetBits(cf0.result_bit_vector_h[i]);
			total_matches += line_matches;
			printf("accepting %dth -- %08x / match# : %d\n", (i%((cf0.accepting_states_count)*cf0.threads_per_block))/cf0.threads_per_block, cf0.result_bit_vector_h[i], line_matches);
		}
		printf("total_matches = %d\n",total_matches);
#else
		
		int global_id=0;
		for( int i=0;i<cf0.nfa_size; i++){
			printf("state %ds -- ",i);
			for(int j =0; j< cf0.threads_per_block;j++){
				printf("%08x ", cf0.result_bit_vector_h[global_id]);
				global_id++;
			}
			printf("\n");
		}
#endif
	}
	 if (strcmp(cf0.application,"Hamming")== 0){//M:hamming

		//M: it will show only block 0 content

#ifdef STATE_VECTOR_DEBUG

	printf("\nPPPPPFFFFFF\n");
		//hamming distance validate/print
		unsigned buffer[30][30] = {0};//used to flip output so it is easier to read
		int i,j,p;
		int depth = 1;
		int counter = 0;
		int w = 0;
		for(w=0; w<cf0.threads_per_block; w++){
			printf("\nThread %d's workload------\n",w);

			int loc = 0;
			for(i=0; i<cf0.k+1; i++){
				for(j=0; j<min(i+1, cf0.d+1); j++){
					for(p=0; p<cf0.threads_per_block; p++){
						if(p==w){
							buffer[i][j] = cf0.result_bit_vector_h[  (loc)*cf0.threads_per_block   +   j*cf0.threads_per_block   +   p];
						}
					}
				}
				loc+=min(i+1,cf0.d+1);
			}

			for(j=0; j<cf0.d+1; j++){
				for(i=j ; i<cf0.k+1; i++){
					printf("%08x ", buffer[i][j]);
				}
				printf("\n");
				for(p=0; p<j+1; p++){
					printf("         ");
				}
			}
			printf("\n----------()----------------\n");
		}
#else

		unsigned i = 0;
		unsigned j = 0;
		unsigned k = 0;
		unsigned l = 0;

		for(i=0; i < (cf0.accepting_states_count)*cf0.blocks_count*cf0.threads_per_block*cf0.packets_per_kernel_launch; i++){
			line_matches = numberOfSetBits(cf0.result_bit_vector_h[i]);
			total_matches += line_matches;
			printf("D=%d -- %08x / match# : %d\n", i%(cf0.accepting_states_count), cf0.result_bit_vector_h[i],line_matches);
		}
		printf("total_matches = %d\n",total_matches);
	#endif
	}
	 if (strcmp(cf0.application,"Levenshtein")== 0){//M:levenshtein
#ifdef STATE_VECTOR_DEBUG

		int i,j,p;
		int w = 0;
		for(w=0; w<cf0.threads_per_block; w++){
			printf("\nThread %d's workload------\n",w);

			for(i=0; i< (cf0.d+1); i++){	//i shows which row we are printing at now
				for(j=i; j< i+(cf0.k+1)*(cf0.d+1); j+=(cf0.d+1) ){ //j holds state id
					for(p=0; p<cf0.threads_per_block; p++){
						if(p==w){	//only print the p(ed) NFA chunk for example for thread 0 only first 32 bit chunk of state vector will be shown
							printf("%08x ", cf0.result_bit_vector_h[j*cf0.threads_per_block  + p]);//M: why is %08x? because its printing * hex digits which shows 32 bits that are 32 NFAstates
						}
					}
				}
				printf("\n");
			}
			printf("--------------------------\n");
		}
#else
		unsigned i = 0;
		unsigned j = 0;
		unsigned k = 0;
		unsigned l = 0;

		for(i=0; i < (cf0.accepting_states_count)*cf0.threads_per_block*cf0.blocks_count*cf0.packets_per_kernel_launch; i++){
			line_matches = numberOfSetBits(cf0.result_bit_vector_h[i]);
			total_matches += line_matches;
			printf("accepting %dth -- %08x / match# : %d\n", (i%((cf0.accepting_states_count)*cf0.threads_per_block))/cf0.threads_per_block, cf0.result_bit_vector_h[i],line_matches);
		}
		printf("total_matches = %d\n",total_matches);
#endif
	}

#endif
	free(cf0.result_bit_vector_h);
	return 0;
}

/*
 *  MAIN - entry point
 */
int main(int argc, char **argv){

		parse_arguments(argc,argv);

		general_initialize();

		general_preprocessed_to_device();

		general_inputstreaming();

	return 0;
}

/* parse the main call parameters */
static int parse_arguments(int argc, char **argv)
{
	int i=1;
    if (argc < 2) {
       // usage();
		printf("arguments numver wrong\n");
		return 0;
    }
    while(i<argc){
    	if(strcmp(argv[i], "-af") == 0 || strcmp(argv[i], "--automata_file") == 0){
    		i++;
    		if(i==argc){
    			fprintf(stderr," Automata transition file base name missing!\n");
    			return 0;
    		}
    	//	cf0.compiler_output_filename = argv[i];
    		strcpy(cf0.compiler_output_filename ,argv[i]);
	}else if(strcmp(argv[i], "-tnum") == 0 || strcmp(argv[i], "--trace_num") == 0){
    		i++;
    		if(i==argc){
    			fprintf(stderr,"Number of trace files missing.\n");
    			return 0;
    		}
    		cf0.trace_num= atoi(argv[i]);
		i++;
		if(strcmp(argv[i], "-tnames") == 0 || strcmp(argv[i], "--tracefile_names") == 0){
			i++;
    			if(i==argc){
    				fprintf(stderr,"Name of trace files missing.\n");
    				return 0;
    			}
			//char **temp= new char *[config.get_trace_num()];
			cf0.stream_sequence_filename= new char *[cf0.trace_num];
			char **temp = cf0.stream_sequence_filename;
			for(int j = 0; j < cf0.trace_num; j++){ 
    				temp[j]=argv[i]; i++;//TODO: check if its correct
			}
			i --;
		}
		else{
			fprintf(stderr,"Name of trace files should follow number of traces.\n");
    			return 0;
		}

    	}else if(strcmp(argv[i], "-pn") == 0 || strcmp(argv[i], "--pkt_num") == 0){
    		i++;
    		if(i==argc){
    			fprintf(stderr," number of packets in each kernel call missing!\n");
    			return 0;
    		}
    		cf0.packets_per_kernel_launch= atoi(argv[i]);
    	}else if(strcmp(argv[i], "-ps") == 0 || strcmp(argv[i], "--pkt_size") == 0){
    		i++;
    		if(i==argc){
    			fprintf(stderr," size of packets missing!\n");
    			return 0;
    		}
    		cf0.packet_size=atoi(argv[i]);
    	}else if(strcmp(argv[i], "-tn") == 0 || strcmp(argv[i], "--thread_num") == 0){
    		i++;
    		if(i==argc){
    			fprintf(stderr," number of threads missing!\n");
    			return 0;
    		}
    		cf0.threads_per_block= atoi(argv[i]);
    	}else if(strcmp(argv[i], "-dev") == 0 || strcmp(argv[i], "--device") == 0){
    		i++;
    		if(i==argc){
    			fprintf(stderr," device ID missing!\n");
    			return 0;
    		}
    		cf0.gpu_device=atoi(argv[i]);
    	}else if(strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--distance") == 0){
    		i++;
    		if(i==argc){
    			fprintf(stderr," hamming or levenshtein distance missing!\n");
    			return 0;
    		}
    		cf0.d=atoi(argv[i]);
    	}else if(strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--kmer_size") == 0){
    		i++;
    		if(i==argc){
    			fprintf(stderr," size of k-mer missing!\n");
    			return 0;
    		}
    		cf0.k=atoi(argv[i]);
    	}else if(strcmp(argv[i], "--stream_count") == 0 ){
    		i++;
    		if(i==argc){
    			fprintf(stderr," cf0.occupancy_efficient_stream_count missing!\n");
    			return 0;
    		}
    		stream_count_from_command_line=atoi(argv[i]);
    	}else{
    		fprintf(stderr,"Ignoring invalid option %s\n",argv[i]);
    	}
    	i++;
    }
	return 1;
}
