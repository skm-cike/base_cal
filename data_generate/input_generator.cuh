//
// Created by fil on 24-9-19.
//

#ifndef SNARKVM_CUDA_INPUT_GENERATOR_CUH
#define SNARKVM_CUDA_INPUT_GENERATOR_CUH
#include <curand_kernel.h>
#include "../base_cal/bigint_field256.cuh"
namespace data_gen {
    // 生成随机 bool 值的设备端函数
    __device__ bool genRanBool(curandState *state);

    // 生成随机 uint8_t 数的设备端函数
    __device__ I8 genRan8(curandState *state);

    // 生成随机 uint16_t 数的设备端函数
    __device__ I16 genRan16(curandState *state);

    __device__ I32 genRan32(curandState *state);

    // 生成随机 int64 数的设备端函数
    __device__ I64 genRan64(curandState *state);

    // 生成随机 128位 随机数
    __device__ skm::BigInt* genRan128(curandState *state);

    // 生成随机 128位 随机数
    __device__ skm::BigInt* genRanField(curandState *state);
}

#endif //SNARKVM_CUDA_INPUT_GENERATOR_CUH
