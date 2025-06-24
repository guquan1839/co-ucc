// #ifndef COMPRESS_NCCL_H
// #define COMPRESS_NCCL_H


// extern "C"{

// #include "nccl.h"
// #include "compresskernel.h"
// #include "get_minmax.h"
// #include "base_tool.h"
// #include "cuda_checks.h"

// ncclResult_t ncclCompress();

// ncclResult_t ncclDecompress();

// ncclResult_t ncclDecompressReduce();

// }

// #endif

#ifndef COMPRESS_NCCL_H
#define COMPRESS_NCCL_H


#include "nccl.h"
#include "compresskernel.h"
#include "get_minmax.h"
#include "base_tool.h"
#include "cuda_checks.h"

#ifdef __cplusplus
extern "C" {
#endif



    ncclResult_t ncclCompress(const void* input_arr, const size_t input_chunk_count, ncclDataType_t input_datatype,
        void** output_arr, size_t* output_chunk_count, ncclDataType_t* output_datatype, const size_t num_chunks, cudaStream_t stream);

    ncclResult_t ncclDecompress(const void* input_arr, const size_t input_chunk_count, ncclDataType_t input_datatype,
        void* output_arr,const size_t output_chunk_count, ncclDataType_t output_datatype, const size_t num_chunks, cudaStream_t stream);

    ncclResult_t ncclDecompressReduce(const void* input_arr, const size_t input_chunk_count, ncclDataType_t input_datatype,
        const void* reduce_input_arr, void* output_arr,const size_t output_chunk_count, ncclDataType_t output_datatype, const size_t num_chunks, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#endif
