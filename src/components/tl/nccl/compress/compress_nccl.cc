#include "compress_nccl.h"
#include "nccl.h"
#include <stdio.h>
#include "cuda_checks.h"
#include <cstdint>

extern "C"{
    ncclResult_t launch_compress(const void* input_arr, const size_t input_chunk_count, ncclDataType_t input_datatype,
    void* output_arr, const size_t  output_chunk_count, ncclDataType_t output_datatype, const size_t num_chunks, cudaStream_t stream);

    ncclResult_t launch_decompress(const void* input_arr, const size_t input_chunk_count, ncclDataType_t input_datatype,
    void* output_arr, const size_t output_chunk_count, ncclDataType_t output_datatype, const size_t num_chunks, cudaStream_t stream);
    
    ncclResult_t launch_decompress_reduce(const void* input_arr, const size_t input_chunk_count, ncclDataType_t input_datatype,
    const void* reduce_input_arr, void* output_arr, const size_t output_chunk_count, ncclDataType_t output_datatype, const size_t num_chunks, cudaStream_t stream);

    ncclResult_t ncclCompress(const void* input_arr, const size_t input_chunk_count, ncclDataType_t input_datatype,
    void** output_arr, size_t *output_chunk_count, ncclDataType_t* output_datatype, const size_t num_chunks, cudaStream_t stream){

    //base min-max uint 8

        *output_datatype = ncclDataType_t::ncclUint8;

        if(input_datatype == ncclDataType_t::ncclFloat32){
            *output_chunk_count = (ucc_compress_align(input_chunk_count, 32) + ucc_compress_align(4 * 2, 32));

        }
        else if(input_datatype == ncclDataType_t::ncclFloat16){
            *output_chunk_count = (ucc_compress_align(input_chunk_count, 32) + ucc_compress_align(2 * 2, 32));

        }

        if(*output_arr == nullptr){
            CUDACHECK(cudaMallocAsync((void**)output_arr,(*output_chunk_count) * num_chunks * ncclTypeSize(*output_datatype),stream));

        }

        NCCLCHECK(launch_compress(input_arr,input_chunk_count,input_datatype,*output_arr,*output_chunk_count,*output_datatype,num_chunks,stream));

        return ncclSuccess;

    }


    ncclResult_t ncclDecompress(const void* input_arr, const size_t input_chunk_count, ncclDataType_t input_datatype,
        void* output_arr,const size_t output_chunk_count, ncclDataType_t output_datatype, const size_t num_chunks, cudaStream_t stream){

        NCCLCHECK(launch_decompress(input_arr, input_chunk_count, input_datatype, output_arr, output_chunk_count, output_datatype, num_chunks, stream));

        return ncclSuccess;
    }


    ncclResult_t ncclDecompressReduce(const void* input_arr, const size_t input_chunk_count, ncclDataType_t input_datatype,
        const void* reduce_input_arr, void* output_arr, const size_t output_chunk_count, ncclDataType_t output_datatype, const size_t num_chunks, cudaStream_t stream){

        NCCLCHECK(launch_decompress_reduce(input_arr, input_chunk_count, input_datatype, reduce_input_arr, output_arr, output_chunk_count, output_datatype, num_chunks, stream));

        return ncclSuccess;
}
}