//
// Created by fil on 24-9-26.
//

#ifndef SNARKVM_CUDA_POSEIDON_ARK_AND_MDS_CUH
#define SNARKVM_CUDA_POSEIDON_ARK_AND_MDS_CUH
#include "../base_cal/bigint_field256.cuh"

namespace ark_and_mds_pos {
    struct PoseidonGrainLFSR {
        u64 field_size_in_bits;
        bool state[80];
        u64 head;
        __device__ bool next_bit();
        __device__ static PoseidonGrainLFSR* new_(bool is_sbox_an_inverse, u64 field_size_in_bits, u64 state_len, u64 num_full_rounds, u64 num_partial_rounds);
        __device__ Vec<bool>* get_bits(Vec<bool> *bits, u64 field_size_in_bits);
        __device__ bool get_bit();
        __device__ Vec<skm::BigInt*>* get_field_elements_rejection_sampling(u64 num_elements);
        __device__ Vec<skm::BigInt*>* get_field_elements_mod_p(u64 num_elems);
    };

    struct POSEIDON {
        U64 RATE;
        U64 full_rounds;
        U64 partial_rounds;
        U64 skip_matrices;
        skm::BigInt* domain;
        skm::BigInt* alpha;
        Vec<Vec<skm::BigInt*>*>* ark;
        Vec<Vec<skm::BigInt*>*>* mds;
        __device__ static POSEIDON* new_(const POSEIDON* src);
        __device__ void find_poseidon_ark_and_mds();
        __device__ void serial_batch_inversion_and_mul(Vec<skm::BigInt*>* v, const skm::BigInt* coeff);
    };
}

#endif //SNARKVM_CUDA_POSEIDON_ARK_AND_MDS_CUH
