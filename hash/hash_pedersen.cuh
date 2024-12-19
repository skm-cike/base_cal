//
// Created by fil on 24-9-26.
//

#ifndef SNARKVM_CUDA_HASH_PEDERSEN_CUH
#define SNARKVM_CUDA_HASH_PEDERSEN_CUH
#include "../base_cal/bigint_group.cuh"
namespace hash_pedersen {
    __device__ __constant__ char* DomainAleoPedersen64  = "AleoPedersen64";
    __device__ __constant__ char* DomainAleoPedersen128  = "AleoPedersen128";

    Group* hash_uncompressed() {

    }

    skm::BigInt* hash(Vec<skm::BigInt*> *inputs) {
        Group* group = hash_uncompressed();
        skm::BigInt* rst = group->x;
        delete(group->y);
        delete(group);
        return rst;
    }


}
#endif //SNARKVM_CUDA_HASH_PEDERSEN_CUH
