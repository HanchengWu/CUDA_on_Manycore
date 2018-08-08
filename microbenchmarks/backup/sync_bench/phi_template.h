#ifndef PHI_TEMPLATE
#define PHI_TEMPLATE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>

//Simple MACRO renaming
#define vint  __m512i
#define vfloat  __m512
#define vmask __mmask16

#define ALLOC   alloc_if(1) free_if(0)
#define FREE    alloc_if(0) free_if(1)
#define REUSE   alloc_if(0) free_if(0)

inline double gettime_ms() {
        struct timeval t;
        gettimeofday(&t,NULL);
        return (t.tv_sec+t.tv_usec*1e-6)*1000;
}

#endif

