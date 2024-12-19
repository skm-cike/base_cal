//
// Created by fil on 24-9-26.
//

#ifndef SNARKVM_CUDA_HASH_POSEIDON_CUH
#define SNARKVM_CUDA_HASH_POSEIDON_CUH
#include "../base_cal/bigint.cuh"
#include "poseidon_ark_and_mds.cuh"
namespace hash_poseidon {
    struct Range {
        U64 start;
        U64 end;
        __device__ bool contains(int* i);
    };
    enum DuplexSpongeModeEnum {
        Absorbing,
        Squeezing
    };
    struct DuplexSpongeMode {
        DuplexSpongeModeEnum duplexSpongeMode;
        u64 next_index;
    };


    struct PoseidonDefaultParametersEntry{
        u32 rate;
        u32 alpha;
        u32 full_rounds;
        u32 partial_rounds;
        u32 skip_matrices;
    };

    //===========================重要常数 start=================================
    __device__ __constant__ extern PoseidonDefaultParametersEntry poseidonDefaultParametersFq256[];
    __device__ __constant__ extern u32 poseidonDefaultParametersFq256Size;
    __device__ __constant__ extern U8 CAPACITY;
    __device__ __constant__ extern u32 poseidon2_rate;
    __device__ __constant__ extern u32 poseidon4_rate;
    __device__ __constant__ extern u32 poseidon8_rate;
    __device__ __constant__ extern char* DomainAleoPoseidon2;
    __device__ __constant__ extern char* DomainAleoPoseidon4;
    __device__ __constant__ extern char* DomainAleoPoseidon8;
    //===========================重要常数 end=================================

    //全局变量
    __device__ extern ark_and_mds_pos::POSEIDON *POSEIDON_2;

    __device__ skm::BigInt *hash_psd2(Vec<skm::BigInt *> *inputs);
}


#endif //SNARKVM_CUDA_HASH_POSEIDON_CUH
