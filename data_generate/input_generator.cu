//
// Created by fil on 24-10-16.
//
#include "input_generator.cuh"
namespace data_gen {
    // 生成随机 bool 值的设备端函数
    __device__ bool genRanBool(curandState *state) {
        // 生成一个 uint32_t 随机数
        I32 rand32 = curand(state);
        // 将 uint32_t 随机数映射到布尔值
        // 假设我们用一半的概率返回 true 或 false
        return (rand32 % 2 == 0); // 随机选择 0 或 1
    }

    // 生成随机 uint8_t 数的设备端函数
    __device__ I8 genRan8(curandState *state) {
        // 生成一个 uint32_t 随机数
        I32 rand32 = curand(state);

        // 将 uint32_t 转换为 int8_t
        // 生成的数在 -128 到 127 之间
        return static_cast<I8>(rand32 % 256 - 128); // 256 = 2^8, 范围 -128 到 127
    }

    // 生成随机 uint16_t 数的设备端函数
    __device__ I16 genRan16(curandState *state) {
        // 生成一个 uint32_t 随机数
        I32 rand32 = curand(state);
        // 将 uint32_t 转换为 int16_t
        // 生成的数在 -32768 到 32767 之间
        return static_cast<I16>(rand32 % 65536 - 32768); // 65536 = 2^16, 范围 -327
    }

    __device__ I32 genRan32(curandState *state) {
        return curand(&state[0]);
    }

    // 生成随机 int64 数的设备端函数
    __device__ I64 genRan64(curandState *state) {
        // 生成两个 uint32 数，并将它们合并成一个 uint64 数
        I32 low = curand(&state[0]);
        I32 high = curand(&state[0]);
        return (static_cast<I64>(high) << 32) | low;
    }

    // 生成随机 128位 随机数
    __device__ skm::BigInt* genRan128(curandState *state) {
        U64 low = genRan64(state);
        U64 high = genRan64(state);
        return skm::BigIntOper::CreatInt128(low, high, true);
    }

    // 生成随机 128位 随机数
    __device__ skm::BigInt* genRanField(curandState *state) {
        while (true) {
            U64 l1 = genRan64(state);
            U64 l2 = genRan64(state);
            U64 l3 = genRan64(state);
            U64 high = genRan64(state);
            high &= (MAX_U64 >> 3);
            skm::BigInt *rst = skm::BigIntOper::CreatInt256(l1, l2, l3, high, false);
            if (skm::BigIntOperField256::is_valid(rst)) {
                return rst;
            }
            skm::BigIntOper::destory(rst);
        }
    }
}