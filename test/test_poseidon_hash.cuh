//
// Created by fil on 24-9-30.
//

#ifndef SNARKVM_CUDA_TEST_POSEIDON_HASH_CUH
#define SNARKVM_CUDA_TEST_POSEIDON_HASH_CUH
//#include "../hash/poseidon_ark_and_mds.cuh"
#include "../hash/hash_poseidon.cuh"
#include "../base_cal/bigint_field256.cuh"
#include "../base_cal/cache/VmCache.cuh"
namespace test_poseidon {
    __device__  unsigned long long index = 0;

    //ok
    __device__ void test_psd2() {
        Vec<skm::BigInt *> *inputs = VmCache::getBigIntLst(2);
        skm::BigInt *i1 = skm::BigIntOperField256::CreatInt256(0x45C089FB7CE554A5, 0x1D58AF1B3E0F963F,
                                                               0x5D0D28BD69A0822D, 0x90BE533BE7F84E4, false);

        skm::BigInt *i2 = skm::BigIntOperField256::CreatInt256(0xF09FFFF73ECF0DC, 0x3864E5FA2E6A9EF7,
                                                               0x827113F0F7B84AD4, 0x129F57EE31BCE8A1, false);
        inputs->push(i1);
        inputs->push(i2);

        while(true) {
            skm::BigInt *a = hash_poseidon::hash_psd2(inputs);

            u64 c = atomicAdd(&index, 1ull);
            if (c % 300 == 0) {
                printf("--------%lld\n", index);
            }
            VmCache::returnBigInt(a);
            break;
        }

//        skm::BigInt *a = hash_poseidon::hash_psd2(inputs);
//        skm::BigIntOperField256::print_hex(a);
//        a = hash_poseidon::hash_psd2(inputs);
//        skm::BigIntOperField256::print_hex(a);
    }


}

#endif //SNARKVM_CUDA_TEST_POSEIDON_HASH_CUH
