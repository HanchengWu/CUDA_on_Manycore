#ifndef __TIMING_H__
#define __TIMING_H__

#include <sys/time.h>

inline double gettime_ms() {
        struct timeval t;
        gettimeofday(&t,NULL);
        return (t.tv_sec+t.tv_usec*1e-6)*1000;
}

#endif
