#include "cuda_runtime.h"
#include "nccl.h"
// #include <cstdio>

#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(err));  \
        return ncclUnhandledCudaError;                      \
    }                                                       \
} while(0)


#define NCCLCHECK(cmd) do {                                  \
    ncclResult_t res = (cmd);                                \
    if (res != ncclSuccess) {                                \
        fprintf(stderr, "NCCL error: %s", ncclGetErrorString(res)); \
        return res; \
    }                                                         \
} while(0)
