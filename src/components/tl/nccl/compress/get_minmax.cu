#include "nccl.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <limits>
#include <cstdint>

int cal_grid_num(int data_size, int block_size){
    return (data_size + block_size - 1) / block_size;
}

template<typename Datatype>
__device__ inline Datatype compress_add(Datatype value_a, Datatype value_b){
    return value_a + value_b;
} 

template<>
__device__ inline half compress_add(half value_a,half value_b){
    return __hadd(value_a,value_b);
}



template<typename Datatype>
__device__ inline void store_data(Datatype *arr, float data){
    arr[0] = data;
}

template<>
__device__ inline void store_data<half>(half *arr, float data){
    arr[0] = __float2half(data);
}


// template<typename Datatype>
__device__ static float atomic_max(float *address, float value){

    int* original_value = (int*)address;
    int old_value = *original_value;
    int assumed;


    do{
        assumed = old_value;
        old_value = :: atomicCAS(original_value, assumed, __float_as_int(::fmaxf(value, __int_as_float(assumed))));

    }while(assumed != old_value);


    return __int_as_float(old_value);
}



__device__ static float atomic_min(float *address, float value){
    int* original_value = (int*) address;
    int old_value = *original_value;
    int assumed;

    do{
        assumed = old_value;
        old_value = :: atomicCAS(original_value, assumed, __float_as_int(::fminf(value, __int_as_float(assumed))));

    }while(assumed != old_value);

    return __int_as_float(old_value);
}

template<typename Datatype>
__device__ inline Datatype get_infinity();

template<>
__device__ inline float get_infinity<float>(){
    return INFINITY;
}

template<>
__device__ inline half get_infinity<half>(){
    return __float2half(INFINITY);
}


template<typename Datatype>
__device__ void warp_reduce(volatile Datatype* min_val, volatile Datatype* max_val, const int tid, const int block_size){
    if(block_size >= 64){
        min_val[tid] = (min_val[tid] < min_val[tid + 32]) ? min_val[tid] : min_val[tid + 32];
        max_val[tid] = (max_val[tid] > max_val[tid + 32]) ? max_val[tid] : max_val[tid + 32];

    }

    if(block_size >= 32){
        min_val[tid] = (min_val[tid] < min_val[tid + 16]) ? min_val[tid] : min_val[tid + 16];
        max_val[tid] = (max_val[tid] > max_val[tid + 16]) ? max_val[tid] : max_val[tid + 16];

    }

    if(block_size >= 16){
        min_val[tid] = (min_val[tid] < min_val[tid + 8]) ? min_val[tid] : min_val[tid + 8];
        max_val[tid] = (max_val[tid] > max_val[tid + 8]) ? max_val[tid] : max_val[tid + 8];

    }

    if(block_size >= 8){
        min_val[tid] = (min_val[tid] < min_val[tid + 4]) ? min_val[tid] : min_val[tid + 4];
        max_val[tid] = (max_val[tid] > max_val[tid + 4]) ? max_val[tid] : max_val[tid + 4];

    }

    if(block_size >= 4){
        min_val[tid] = (min_val[tid] < min_val[tid + 2]) ? min_val[tid] : min_val[tid + 2];
        max_val[tid] = (max_val[tid] > max_val[tid + 2]) ? max_val[tid] : max_val[tid + 2];

    }

    if(block_size >= 2){
        min_val[tid] = (min_val[tid] < min_val[tid + 1]) ? min_val[tid] : min_val[tid + 1];
        max_val[tid] = (max_val[tid] > max_val[tid + 1]) ? max_val[tid] : max_val[tid + 1];

    }

}


template <typename Datatype>
__global__ void minmax_blockreduce(const void* input_arr, const size_t input_chunk_count, void* output_arr, const size_t output_chunk_count){
    extern __shared__ unsigned char mem[];

    Datatype* shared_mem = reinterpret_cast<Datatype*>(mem);

    Datatype* input_buffer = (Datatype*) input_arr;
    uint8_t* output_buffer = (uint8_t*) output_arr;

    Datatype* shared_Min = shared_mem;
    Datatype* shared_Max = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    int index_i = threadIdx.x + blockIdx.x * blockDim.x;
    int index_j = threadIdx.y + blockIdx.y * blockDim.y;

    const int block_size = blockDim.x;


    Datatype local_min_value = get_infinity<Datatype>();
    Datatype local_max_value = -get_infinity<Datatype>();

    while(index_i < input_chunk_count){
        int k = index_j * input_chunk_count;
        local_max_value = (local_max_value > input_buffer[k + index_i]) ? local_max_value : input_buffer[k + index_i];
        local_min_value = (local_min_value > input_buffer[k + index_i]) ? local_min_value : input_buffer[k + index_i];
        index_i += blockDim.x * gridDim.x;
    }

    shared_Max[tid] = local_max_value;
    shared_Min[tid] = local_min_value;


    __syncthreads();

    //这部分应该是可以删除的，因为block_size = 1024已经是最大值了！
    if(block_size >= 1024){
        if(tid < 512){
            shared_Max[tid] = local_max_value = (local_max_value > shared_Max[tid + 512]) ? local_max_value : shared_Max[tid + 512];
            shared_Min[tid] = local_min_value = (local_min_value < shared_Max[tid + 512]) ? local_min_value : shared_Min[tid + 512];
        }
        __syncthreads();
    }


    if(block_size >= 512){
        if(tid < 256){
            shared_Max[tid] = local_max_value = (local_max_value > shared_Max[tid + 256]) ? local_max_value : shared_Max[tid + 256];
            shared_Min[tid] = local_min_value = (local_min_value < shared_Min[tid + 256]) ? local_min_value : shared_Min[tid + 256];
        }
        __syncthreads();
    }


    if(block_size >= 256){
        if(tid < 128){
            shared_Max[tid] = local_max_value = (local_max_value > shared_Max[tid + 128]) ? local_max_value : shared_Max[tid + 128];
            shared_Min[tid] = local_min_value = (local_min_value < shared_Min[tid + 128]) ? local_min_value : shared_Min[tid + 128];
        }
        __syncthreads();
    }

    if(block_size >= 128){
        if(tid < 64){
            shared_Max[tid] = local_max_value = (local_max_value > shared_Max[tid + 64]) ? local_max_value : shared_Max[tid + 64];
            shared_Min[tid] = local_min_value = (local_min_value < shared_Min[tid + 64]) ? local_min_value : shared_Min[tid + 64];
        }
        __syncthreads();
    }

    if(tid < 32){
        warp_reduce<Datatype>(shared_Min,shared_Max,tid,block_size);
    }

    if(tid == 0){
        Datatype* min_val = reinterpret_cast<Datatype*>(output_buffer + index_j * output_chunk_count);
        Datatype* max_val = reinterpret_cast<Datatype*>(output_buffer + index_j * output_chunk_count + sizeof(Datatype));

        atomic_min(min_val, shared_Min[0]);
        atomic_max(max_val, shared_Max[0]);
    }

}

template <typename Datatype>
__global__ void init_minmax2buffer(void* input_arr, const size_t chunk_count, const size_t chunk_number){

    int index_i = threadIdx.x + blockIdx.x * blockDim.x;
    uint8_t* input_buffer = (uint8_t*) input_arr;

    if(index_i < chunk_number * chunk_count){
        store_data(reinterpret_cast<Datatype*>(input_buffer + index_i * chunk_count), get_infinity<Datatype>());
        store_data(reinterpret_cast<Datatype*>(input_buffer + index_i * chunk_count + sizeof(Datatype)), -get_infinity<Datatype>());
    }
}



void add_minmax2buffer(const void* input_arr, const size_t input_chunk_count, void* output_arr, const size_t output_chunk_count,
    const size_t chunk_number, ncclDataType_t datatype, cudaStream_t stream){

    //考虑一个问题，为什么要init_block和block要单独指定，有区别没？
    int init_block = (chunk_number < 1024) ? chunk_number : 1024;
    int init_grid = cal_grid_num(chunk_number, init_block);

    int block = (input_chunk_count < 1024) ? input_chunk_count : 1024;


    dim3 grid(cal_grid_num(input_chunk_count, block),chunk_number);

    if(datatype == ncclDataType_t::ncclFloat16){

    }
    else if(datatype == ncclDataType_t::ncclFloat32){
        //做了一个初始化
        init_minmax2buffer<float><<<init_grid, init_block, 0, stream>>>(output_arr, output_chunk_count,chunk_number );
        //这里为什么不是0？----》因为使用了动态共享内存
        minmax_blockreduce<float><<<grid, block, 2 * block * sizeof(float), stream>>>(input_arr, input_chunk_count, output_arr, output_chunk_count);
    }

}