#ifndef CUDAERROR_FUNCTION
#define CUDAERROR_FUNCTION

inline void cudaCheckError(cudaError_t ce, unsigned line_number, const char * file)
{
    if (ce != cudaSuccess) {
        printf("CUDA Error: \"%s\", Line Number: %d, File: %s\n\n", cudaGetErrorString(ce), line_number, file);
        exit(1);
     }
}

#endif //CUDAERROR_FUNCTION

