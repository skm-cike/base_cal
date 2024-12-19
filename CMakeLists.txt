cmake_minimum_required(VERSION 3.28)
project(snarkvm_cuda CUDA)

set(CUDA_NVCC_FLAGS -std=c++17 --expt-relaxed-constexpr --device-int128)
set(CMAKE_CUDA_STANDARD 17)
#set(CUDA_NVCC_FLAGS " -std=c++17 --device-int128 -Min128 --expt-relaxed-constexpr")

add_executable(snarkvm_cuda
        base_cal/cache/VmCache.cu
        hash/hash_poseidon.cu
        hash/poseidon_ark_and_mds.cu
        data_generate/input_generator.cu
        base_cal/bigint_field256.cu
        base_cal/bigint.cu
        base_cal/base_type.cuh
        base_cal/bigint_group.cu
        base_cal/basic_oper.cu
        base_cal/base_vec.cu
        base_cal/cache/VmCacheVecOper.cu
        key_gen/variables_gen.cu
        base_cal/base_type.cu
        main.cu
)

set_target_properties(snarkvm_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)