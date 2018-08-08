#ifndef PHI_TEMPLATE_H
#define PHI_TEMPLATE_H
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef PHI_TEMPLATE_CPP
#define PHI_TEMPLATE_CPP
extern unsigned DEV_NUM;
#endif

enum phiMemcpyKind {CpuToPhi=0, PhiToCpu=1};

void init_phi(void);
int phiSetDevice(int device);
int phiGetDevice(void);
//Simple MACRO renaming
#define vint  __m512i
#define vfloat  __m512
#define vmask __mmask16

#define vseti(x) (_mm512_set1_epi32(x))
#define vsetf(x) (_mm512_set1_ps(x))

#define ASIZE 64
#define ALIGN align(ASIZE)

//Used for deviceMemory pointers allocated by cudaMalloc
#define ALLOC_A   alloc_if(1) free_if(0) ALIGN targetptr
#define FREE_A    alloc_if(0) free_if(1) ALIGN targetptr
#define REUSE_A   alloc_if(0) free_if(0) ALIGN targetptr

//scalar parameters passed to kernel during kernel launch 
#define ONCE   alloc_if(1) free_if(1) 

class dim3 {
	public:
	unsigned x;unsigned y;unsigned z;
	
	//constructor
	dim3(void){
		x=0;y=0;z=0;
	}
	dim3(int a){
		x=a;y=1;z=1;
	}
	dim3(int a, int b){
		x=a;y=b;z=1;
	}
	dim3(int a, int b, int c){
		x=a;y=b;z=c;
	}
	dim3(unsigned a){
		x=a;y=1;z=1;
	}
	dim3(unsigned a, unsigned b){
		x=a;y=b;z=1;
	}
	dim3(unsigned a, unsigned b, unsigned c){
		x=a;y=b;z=c;
	}

	//copy constructor
	dim3 (dim3 &old){
		this->x=old.x; this->y=old.y; this->z=old.z;
	}

	void printDim3(void);

};

template <class T> 
inline int phiMalloc(T **devPtr, unsigned size){
    T *tmp = *devPtr;

#ifndef PROCESSOR
    #pragma offload_transfer target(mic: DEV_NUM) nocopy(tmp : length(size) ALLOC_A) 
#endif

    *devPtr = tmp;
    return EXIT_SUCCESS;
}

template <class T> 
inline int phiMemcpy(T *des, T *src , unsigned size, phiMemcpyKind cpytype){

	switch (cpytype){
		case CpuToPhi:
				#ifndef PROCESSOR
				#pragma offload_transfer target(mic: DEV_NUM) in( src[0:size] : into (des[0:size]) REUSE_A ) 
				#endif
				
				break;
		case PhiToCpu:
				#ifndef PROCESSOR
				#pragma offload_transfer target(mic: DEV_NUM) out( src[0:size] : into (des[0:size]) REUSE_A ) 
				#endif
				
				break;
		default :
				printf("\nError: Memory copytype is undefined!\n");
				return EXIT_FAILURE;
				break;
	}
	return EXIT_SUCCESS;
}

template <class T> 
inline int phiFree(T *devPtr){
	#ifndef PROCESSOR
	#pragma offload_transfer target(mic: DEV_NUM) nocopy( devPtr : FREE_A ) 
	#endif

	return EXIT_SUCCESS;
}

#ifndef PROCESSOR
#pragma offload_attribute (push, target(mic))
#endif
typedef struct {
	unsigned x;
	unsigned y;
	unsigned z;
} Dim3;

inline double gettime_ms() {
        struct timeval t;
        gettimeofday(&t,NULL);
        return (t.tv_sec+t.tv_usec*1e-6)*1000;
}


inline void set_self_affixed_to_spare_core(unsigned sparecore){
        cpu_set_t phi_set;//phi_set used to set pthread affinity
        CPU_ZERO(&phi_set); CPU_SET(sparecore,&phi_set);
        pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t), &phi_set);
}
#ifndef PROCESSOR
#pragma offload_attribute (pop)
#endif

#endif

