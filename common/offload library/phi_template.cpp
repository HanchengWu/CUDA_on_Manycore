#define PHI_TEMPLATE_CPP
#include "phi_template.h"

unsigned DEV_NUM = 1;

void dim3::printDim3(void){
	printf("\nDimx = %d, Dimy = %d, Dimz = %d.\n", x, y, z);
}

void init_phi(void){
	#ifndef PROCESSOR
	#pragma offload target(mic: DEV_NUM) 	
	{
		printf("\nInitialize PHI\n");
		fflush(0);//flush the buffer on Xeon Phi in time
	}
	#endif
}

int phiSetDevice(int device){
	DEV_NUM=device;
	return EXIT_SUCCESS;
}

int phiGetDevice(void){
	return DEV_NUM;
}





