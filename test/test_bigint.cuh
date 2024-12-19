//
// Created by fil on 24-10-15.
//

#ifndef SNARKVM_CUDA_TEST_BIGINT_CUH
#define SNARKVM_CUDA_TEST_BIGINT_CUH
#include "../base_cal/bigint_field256.cuh"
namespace  test_bigint {
    __device__ void test_div_w() {
        skm::BigInt* a = skm::BigIntOper::CreatInt128(0xD0255A280CCD98F9,0x539ADBA05C804907, false);
        skm::BigInt* b = skm::BigIntOper::CreatInt128(0xE11DB69290B2C94F, 0xD6389607C3F346,false);
        skm::BigInt* c = skm::BigIntOper::div_w(a, b);
        skm::BigIntOper::print_hex(a);
        skm::BigIntOper::print_hex(b);
        skm::BigIntOper::print_hex(c);
    }

    //ok
    __device__ void test_shl_w() {
        skm::BigInt* a = skm::BigIntOper::CreatInt128(0xD0255A280CCD98F9,0x539ADBA05C804907, false);
        u64 index = 0;
        while (true) {
            skm::BigIntOper::destory(skm::BigIntOper::shl_w(a,125));
            ++index;
            if (index % 100000 == 0 && index != 0) {
                printf("%lld \n", index);
            }
        }
//        skm::BigIntOper::print_hex(skm::BigIntOper::shl_w(a,125));
//        skm::BigIntOper::print_hex(skm::BigIntOper::shl_w(a,126));
//        skm::BigIntOper::print_hex(skm::BigIntOper::shl_w(a,127));
//        skm::BigIntOper::print_hex(skm::BigIntOper::shl_w(a,128));
//        skm::BigIntOper::print_hex(skm::BigIntOper::shl_w(a,129));
//        skm::BigIntOper::print_hex(skm::BigIntOper::shl_w(a,63));
//        skm::BigIntOper::print_hex(skm::BigIntOper::shl_w(a,64));
//        skm::BigIntOper::print_hex(skm::BigIntOper::shl_w(a,65));
//        skm::BigIntOper::print_hex(skm::BigIntOper::shl_w(a,1));
    }
    //ok
    __device__ void test_shr_w() {
        skm::BigInt* a = skm::BigIntOper::CreatInt128(0xD0255A280CCD98F9,0x539ADBA05C804907, false);
        skm::BigIntOper::print_hex(skm::BigIntOper::shr_w(a,1));
        skm::BigIntOper::print_hex(skm::BigIntOper::shr_w(a,62));
        skm::BigIntOper::print_hex(skm::BigIntOper::shr_w(a,69));
        skm::BigIntOper::print_hex(skm::BigIntOper::shr_w(a,127));
        skm::BigIntOper::print_hex(skm::BigIntOper::shr_w(a,128));
        skm::BigIntOper::print_hex(skm::BigIntOper::shr_w(a,129));

    }
}
#endif //SNARKVM_CUDA_TEST_BIGINT_CUH
