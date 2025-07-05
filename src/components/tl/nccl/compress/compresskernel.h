#include "nccl.h"
#include "get_minmax.h"
#include "cuda_checks.h"

#ifdef __cplusplus
extern "C" {
#endif

    ncclResult_t launch_compress(const void* original_array, const size_t original_chunk_count,ncclDataType_t original_type,void * compressed_array,const size_t compressed_chunk_count,
        ncclDataType_t compressed_type, const size_t chunk_number, cudaStream_t stream);

    ncclResult_t launch_decompress(const void* compressed_array, const size_t compressed_chunk_count,ncclDataType_t compressed_type,void * original_array,const size_t original_chunk_count,
        ncclDataType_t original_type, const size_t chunk_number, cudaStream_t stream);

    ncclResult_t launch_decompress_reduce(const void* compressed_array, const size_t compressed_chunk_count,ncclDataType_t compressed_type,
        const void* reduce_input_array, void* original_array,const size_t original_chunk_count,
        ncclDataType_t original_type, const size_t chunk_number, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
