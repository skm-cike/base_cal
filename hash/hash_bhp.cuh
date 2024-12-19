//
// Created by fil on 24-10-15.
//

#ifndef SNARKVM_CUDA_HASH_BHP_CUH
#define SNARKVM_CUDA_HASH_BHP_CUH
namespace hash_bhp {
    __device__ __constant__ char* DomainAleoBHP256  = "AleoBHP256";
    __device__ __constant__ char* DomainAleoBHP512  = "AleoBHP512";
    __device__ __constant__ char* DomainAleoBHP768  = "AleoBHP768";
    __device__ __constant__ char* DomainAleoBHP1024  = "AleoBHP1024";
}
#endif //SNARKVM_CUDA_HASH_BHP_CUH
