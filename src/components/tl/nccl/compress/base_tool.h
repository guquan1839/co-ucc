#include "nccl.h"
#include <stdio.h>


size_t ucc_compress_align(size_t data_size, size_t block_size){
    return (data_size + block_size - 1) / block_size;
}



inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
    return 1;
  case ncclFloat16:
  #if defined(__CUDA_BF16_TYPES_EXIST__)
  case ncclBfloat16:
  #endif
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}