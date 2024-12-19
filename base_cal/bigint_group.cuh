//
// Created by fil on 24-9-26.
//

#ifndef SNARKVM_CUDA_BIGINT_GROUP_CUH
#define SNARKVM_CUDA_BIGINT_GROUP_CUH
#include "bigint_field256.cuh"
struct Group {
    skm::BigInt* x;
    skm::BigInt* y;
};

#endif //SNARKVM_CUDA_BIGINT_GROUP_CUH
