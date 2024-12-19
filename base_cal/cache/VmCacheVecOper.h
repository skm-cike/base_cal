//
// Created by fil on 24-10-18.
//

#ifndef SNARKVM_CUDA_VMCACHEVECOPER_H
#define SNARKVM_CUDA_VMCACHEVECOPER_H

#include <ostream>
#include "VmCache.cuh"
#include "../base_vec.cuh"
class VmCacheVecOper {
public:
    __device__ static  Vec<u8>* cloneU8(Vec<u8>* vec);
    __device__  static  Vec<Bool>* cloneBoolStruct(Vec<Bool>* vec, int len);
    __device__ static Vec<Vec<Bool>*>* chunksBoolStruct(Vec<Bool>* vec,int chunk_size);
    __device__  static Vec<Vec<bool>*>* chunksBool(Vec<bool>* vec,int chunk_size);
    __device__ static Vec<Vec<skm::BigInt*>*>* chunksBigInt(Vec<skm::BigInt*>* vec,int chunk_size);
    __device__ static Vec<u8>* sub_array_change_u8(Vec<u8>* vec, int start_index);
    __device__ static Vec<skm::BigInt*>* sub_array_change_big_int(Vec<skm::BigInt*>* vec, int start_index);
    __device__ static Vec<Vec<u8>*>* split_at_u8(Vec<u8>* vec, int split_index);
};
#endif //SNARKVM_CUDA_VMCACHEVECOPER_H
