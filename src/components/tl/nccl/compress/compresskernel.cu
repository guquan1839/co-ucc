#include "nccl.h"
// #include "stdio.h"
#include "compresskernel.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "limits.h"
#include <cstdint>
// #include <common.h>
// #include <align.h>
#include "get_minmax.h"
#include <stdio.h>
const float eps = 1e-7;

template<typename Datatype>
__device__ inline Datatype compress_add(Datatype value_a, Datatype value_b){
    return value_a + value_b;
} 

template<>
__device__ inline half compress_add(half value_a,half value_b){
    return __hadd(value_a,value_b);
}

//这有问题
template<typename Datatype>
__device__ inline uint8_t com_val(Datatype input_value,float scale, float lower_bound, float upper_bound){
    float compressed_value = input_value * scale;
    compressed_value = min(compressed_value,upper_bound);
    uint8_t offset = compressed_value - lower_bound;
    return offset;
}

template<>
__device__ inline uint8_t com_val<half>(half input_value,float scale,float lower_bound,float upper_bound){
    //float compressed_value = rintf(input_value *scale);
    float compressed_value = __half2float(input_value) * scale;
    uint8_t offset = compressed_value - lower_bound;
    return offset;
}



template<>
__device__ inline uint8_t com_val<float>(float input_value,float scale,float lower_bound,float upper_bound){
    float compressed_value  = input_value * scale;
    uint8_t offset = compressed_value - lower_bound;
    return offset;
}



template<typename Datatype>
__device__ inline Datatype decom_val(uint8_t compressed_value,float scale, float lower_bound, Datatype dummy){
    return (Datatype)((compressed_value + lower_bound) / scale);
}

template<>
__device__ inline half decom_val<half>(uint8_t compressed_value,float scale, float lower_bound, half dummy){
    return __float2half((compressed_value + lower_bound) / scale);
}

// template<typename Datatype>
// __device__ inline Datatype decom_val(uint8_t compressed_value,float scale, float lower_bound){
//     return (Datatype)((compressed_value + lower_bound) / scale);
// }

// template<>
// __device__ inline half decom_val<half>(uint8_t compressed_value,float scale, float lower_bound){
//     return __float2half((compressed_value + lower_bound) / scale);
// }


// template<typename Datatype>
// __device__ inline uint8_t block_mimax_algo(){

// }

// template<>
// __device__ inline uint8_t block_mimax_algo(){

// }

// template<>
// __device__ inline uint8_t block_mimax_algo(){
    
// }





template<typename Datatype>
__device__ inline float load_data(Datatype *arr){
    return arr[0];
}

template<>
__device__ inline float load_data(half *arr){
    return __half2float(arr[0]);
}



template<typename Datatype>
__device__ inline void store_data(Datatype * arr, float data){
    arr[0] = data;
}

template<>
__device__ inline void store_data<half>(half * arr, float data){
    arr[0] = __float2half(data);
}



template<typename Datatype>
__global__ void compress_float2uint8_kernel(const void * input_arr, int input_chunk_count, void * output_arr, int output_chunk_count){
    Datatype* origin_arr = (Datatype*)input_arr;
    uint8_t* compress_arr = (uint8_t*)output_arr;


    int index_i  = threadIdx.x + blockIdx.x * blockDim.x;
    int index_j  = threadIdx.y + blockIdx.y * blockDim.y;
    
    float min_value,max_value;
    float scale;
    float lower_bound,upper_bound;


    min_value = load_data(reinterpret_cast<Datatype*>(compress_arr + index_j * output_chunk_count));
    max_value = load_data(reinterpret_cast<Datatype*>(compress_arr + index_j * output_chunk_count + sizeof(Datatype)));

    scale = 255.0/(max_value - min_value +eps);
    upper_bound = max_value * scale;
    lower_bound = upper_bound - 255.0;

    for(int m = index_i; m < input_chunk_count; m += blockDim.x * gridDim.x){
        int n = index_j * input_chunk_count + m;
        int k = index_j * output_chunk_count + 32 + m;
        compress_arr[k] = com_val(origin_arr[n], scale,lower_bound,upper_bound);
    }

    if(index_i == 0){
        store_data(reinterpret_cast<Datatype*>(compress_arr + index_j * output_chunk_count),min_value);
        store_data(reinterpret_cast<Datatype*>(compress_arr + index_j * output_chunk_count + sizeof(Datatype)),max_value);
    }


}


template<typename Datatype>
__global__ void decompress_uint82float_kernel(const void* input_arr,int input_chunk_count, void * output_arr, int output_chunk_count){
    uint8_t* compressed_arr = (uint8_t*)input_arr;
    Datatype* decompressed_arr = (Datatype*)output_arr;

    int index_i = threadIdx.x + blockIdx.x * blockDim.x;
    int index_j = threadIdx.y + blockIdx.y * blockDim.y;

    Datatype dummy;
    dummy = 0;

    float min_value,max_value;
    float scale;
    float lower_bound,upper_bound;


    min_value = load_data(reinterpret_cast<Datatype*>(compressed_arr + index_j * input_chunk_count));
    max_value = load_data(reinterpret_cast<Datatype*>(compressed_arr + index_j * input_chunk_count + sizeof(Datatype)));

    scale = 255.0 / (max_value - min_value + eps);
    upper_bound = max_value * scale;
    lower_bound = upper_bound - 255.0;

    for(int m = index_i; m < input_chunk_count; m += blockDim.x * gridDim.x){
        int n = index_j * output_chunk_count + m;
        int k = index_j * input_chunk_count + 32 + m;
        decompressed_arr[n] = decom_val(compressed_arr[k],scale,lower_bound,dummy);
    }

}


template<typename Datatype>
__global__ void decompress_uint82float_reduce_kernel(const void* input_arr, int input_chunk_count, const void* reduce_input_arr, void *output_arr, int output_chunk_count){

    uint8_t* compress_arr = (uint8_t*)input_arr;
    Datatype* decompress_arr = (Datatype*) output_arr;
    Datatype* reduce_buffer = (Datatype*)reduce_input_arr;

    Datatype dummy;
    dummy = 0;

    int index_i = threadIdx.x + blockIdx.x * blockDim.x;
    int index_j = threadIdx.y + blockIdx.y * blockDim.y;

    const float min_value = load_data(reinterpret_cast<Datatype*>(compress_arr + index_j * input_chunk_count));
    const float max_value = load_data(reinterpret_cast<Datatype*>(compress_arr + index_j * input_chunk_count + sizeof(Datatype)));

    float scale = 255.0 / ( max_value - min_value +  eps);
    float upper_bound = rintf(max_value * scale);
    float lower_bound = upper_bound - 255.0;

    for(int m = index_i; m < output_chunk_count; m += blockDim.x * gridDim.x){
        int n = index_j * output_chunk_count + m;
        int k = index_j * input_chunk_count + 32 + m;
        decompress_arr[n] = compress_add(reduce_buffer[n], decom_val(compress_arr[k], scale, lower_bound, dummy));
    }
}


extern "C"{
    ncclResult_t launch_compress(const void* original_array, const size_t original_chunk_count,ncclDataType_t original_type,void * compressed_array,const size_t compressed_chunk_count,
        ncclDataType_t compressed_type, const size_t chunk_number, cudaStream_t stream){

        int block;
        block = (original_chunk_count) < 1024 ? original_chunk_count : 1024;

        dim3 grid(cal_grid_num(original_chunk_count,block),chunk_number);

        if(original_type == ncclDataType_t::ncclFloat16){
            compress_float2uint8_kernel<half><<<grid, block, 0, stream>>>(original_array,original_chunk_count,compressed_array,compressed_chunk_count);
        }
        else if(original_type == ncclDataType_t::ncclFloat32){

            add_minmax2buffer(original_array, original_chunk_count, compressed_array, compressed_chunk_count, chunk_number, original_type, stream);
            //测试
            //这里应该加一个输出！
            //这个输出用来测试压缩的效果
            //printf(compressed_chunk_count[0]);
            compress_float2uint8_kernel<float><<<grid, block, 0, stream>>>(original_array,original_chunk_count,compressed_array,compressed_chunk_count);
        }


        CUDACHECK(cudaGetLastError());

        return ncclSuccess;
    }

    ncclResult_t launch_decompress(const void* compressed_array, const size_t compressed_chunk_count,ncclDataType_t compressed_type,void* decompressed_array,const size_t decompressed_chunk_count,
        ncclDataType_t decompressed_type, const size_t chunk_number, cudaStream_t stream){


        int block;
        block = decompressed_chunk_count < 1024 ? decompressed_chunk_count : 1024;


        dim3 grid(cal_grid_num(decompressed_chunk_count,block),chunk_number);

        if(decompressed_type == ncclDataType_t::ncclFloat16){
            decompress_uint82float_kernel<half><<<grid,block,0,stream>>>(compressed_array, compressed_chunk_count, decompressed_array, decompressed_chunk_count);
        }
        else if(decompressed_type == ncclDataType_t::ncclFloat32){
            decompress_uint82float_kernel<float><<<grid,block,0,stream>>>(compressed_array, compressed_chunk_count, decompressed_array, decompressed_chunk_count);
        }

        CUDACHECK(cudaGetLastError());

        return ncclSuccess;


    }

    ncclResult_t launch_decompress_reduce(const void* compressed_array,const size_t compressed_chunk_count,ncclDataType_t compressed_type,const void* reduce_input_array,void* decompressed_array,const size_t decompressed_chunk_count,
        ncclDataType_t decompressed_type,const size_t chunk_number, cudaStream_t stream){
        int block;
        block = decompressed_chunk_count < 1024 ? decompressed_chunk_count : 1024;

        dim3 grid(cal_grid_num(decompressed_chunk_count,block),chunk_number);

        if(decompressed_type == ncclDataType_t::ncclFloat16){
            decompress_uint82float_reduce_kernel<half><<<grid,block,0,stream>>>(compressed_array,compressed_chunk_count,reduce_input_array,decompressed_array, decompressed_chunk_count);
        }
        else if(decompressed_type == ncclDataType_t::ncclFloat32){
            decompress_uint82float_reduce_kernel<float><<<grid,block,0,stream>>>(compressed_array,compressed_chunk_count,reduce_input_array,decompressed_array, decompressed_chunk_count);
        }

        CUDACHECK(cudaGetLastError());

        return ncclSuccess;
    }
}