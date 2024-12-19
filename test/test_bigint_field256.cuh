//
// Created by fil on 24-10-12.
//

#ifndef SNARKVM_CUDA_TEST_BIGINT_FIELD256_CUH
#define SNARKVM_CUDA_TEST_BIGINT_FIELD256_CUH
#include "../base_cal/bigint_field256.cuh"
namespace test_bingint_field256 {
    //ok
    __device__ void test_field256_pow() {
        skm::BigInt* v = skm::BigIntOperField256::CreatInt256(0x9ec464191dff626dULL, 0xe3afe4fc52de2c3eULL, 0x55098efb31c5bb8aULL, 0x0f51daa50d9eca73ULL, false);
        skm::BigInt* t = skm::BigIntOperField256::CreatInt256(0xd6127fffffffff17, 0x63d9b214afffff0d, 0xfbe5cf5e1150cec5, 0x0200bce5ad5d8461, false);
        u64 index = 0;
        while (true) {
            skm::BigInt *rst = skm::BigIntOperField256::pow(v, t);
            delete(rst);
            index++;
            if (index %10000 == 0 && index!=0) {
               printf("%lld \n", index);
            }
        }
    }

    //ok
    __device__ void test_add_head() {
        Vec<Bool> *vec = Vec<Bool>::init(3, Bool{true, Mode::Public});
        skm::BigIntOperField256::add_head(vec);
        for (int i = 0; i < vec->getSize(); i++) {
           if (vec->get(i).value) {
               printf("true,");
           } else {
               printf("false,");
           }
        }
        delete(vec);
        printf("\n=======================================\n");
        vec = Vec<Bool>::init(3, Bool{false, Mode::Public});
        skm::BigIntOperField256::add_head(vec);
        for (int i = 0; i < vec->getSize(); i++) {
            if (vec->get(i).value) {
                printf("true,");
            } else {
                printf("false,");
            }
        }
    }


    __device__ void test_from_bit_le() {
        printf("11\n");
        Vec<bool>* vec = VmCache::getBoolLst(256);
        for (int i = 0; i < 256; i++) {
           vec->push(true);
        }
        printf("12\n");
        u64 index = 0;
        while (true) {
//            printf("444444\n");
//            skm::BigInt *s = skm::BigIntOperField256::from_bits_le(vec);
//            skm::BigIntOper::destory(s);
//            printf("5555555\n");
            printf("11111\n");
            Vec<Vec<bool>*>* ttt = VmCache::getBoolLstLst(256);
            VmCache::returnBoolLstLst(ttt);
            index++;
            if (index % 100000 == 0) {
                printf("%d\n", index);
            }
            printf("22222\n");
        }
    }
}
#endif //SNARKVM_CUDA_TEST_BIGINT_FIELD256_CUH
