#include "nccl.h"
#include "cuda_checks.h"
#include <cuda_runtime.h>


void add_minmax2buffer(const void* input_arr, const size_t input_chunk_count, void* output_arr, const size_t output_chunk_count,
    const size_t chunk_number, ncclDataType_t datatype, cudaStream_t stream);

int cal_grid_num(int data_size, int block_size);