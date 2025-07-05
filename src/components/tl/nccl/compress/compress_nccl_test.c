// #include "nccl.h"
// #include "compress_nccl.h"
// // #include "argcheck.h"
// c


// NCCL_API(ncclResult_t,ncclAllGatherComp,const void* send_buffer, void* recv_buffer,size_t send_count, 
//     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

// ncclResult_t ncclAllGatherComp(const void* send_buffer, void* recv_buffer,size_t send_count,ncclDataType_t datatype,
//     ncclComm_t comm, cudaStream_t stream){

//     //compress part
//     void* compress_send_buffer = nullptr;
//     void* compress_recv_buffer = nullptr;

//     size_t compress_send_count;
//     ncclDataType_t compress_datatype;

//     CUDACHECK(cudaSetDevice(comm->cudaDev));

//     NCCLCHECK(ncclCompress(send_buffer,send_count,datatype,&compress_send_buffer,&compress_send_count,&compress_datatype,1,stream));

//     CUDACHECK(cudaMallocAsync((void**)&compress_recv_buffer,compress_send_count * comm -> nRanks * ncclTypeSize(compress_datatype),stream));


//     //gather  part
//     struct ncclInfo info = { ncclFuncAllGather,"AllGather",
//         compress_send_buffer, compress_recv_buffer,
//         compress_send_count, compress_datatype,
//         ncclSum,0,comm,stream,
//         ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS
//     };


//     NCCLCHECK(ncclEnqueueCheck(&info));

//     //decompress part
//     NCCLCHECK(ncclDecompress((void*)compress_recv_buffer, compress_send_count, compress_datatype, recv_buffer, send_count, 
//     datatype, comm->nRanks, stream));



//     //free part
//     CUDACHECK(cudaFreeAsync(compress_send_buffer,stream));
//     CUDACHECK(cudaFreeAsync(compress_recv_buffer,stream));
//     return ncclSuccess;

// }


// NCCL_API();
// ncclResult_t ncclAlltoAllComp(){

// }

// // NCCL_API();
// // ncclResult_t ncclAllreduceCompRing(){


// // }


// // NCCL_API();
// // ncclResult_t ncclReduceScatterComp(){

// // }