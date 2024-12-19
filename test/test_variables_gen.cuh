//
// Created by fil on 24-10-24.
//

#ifndef SNARKVM_CUDA_TEST_VARIABLES_GEN_CUH
#define SNARKVM_CUDA_TEST_VARIABLES_GEN_CUH
#include "../base_cal/base_type.cuh"
#include "../base_cal/bigint_field256.cuh"
#include "../base_cal/base_any.cuh"
#include "../key_gen/variables_gen.cuh"
namespace test_variables_gen {
    __device__ void test_public_variable() {
        Vec<Any>* vec = Vec<Any>::init(14);
        vec->push(Any(true));
        vec->push(Any(false));
        vec->push(Any((i8)-88));
        vec->push(Any((i8)126));
        vec->push(Any((i16)4032));
        vec->push(Any((i16)27241));
        vec->push(Any((i32)-760810188));
        vec->push(Any((i32)798113791));
        vec->push(Any((i64)-1913630344837553576));
        vec->push(Any((i64)4368585614685248742));
        skm::BigInt a0,b0;
        a0.size=2;
        a0.is_signed=true;
        b0.size=2;
        b0.is_signed=true;
        a0.number[0]=0x1C9A1F731EC9A8D0;
        a0.number[1]=0x402E9D0D87C0A600;
        b0.number[0]=0x7F1A622F0E58DCE4;
        b0.number[1]=0xC55CFADFA74145FA;
        vec->push(a0);
        vec->push(b0);
        skm::BigInt a,b;
        a.size=4;
        a.is_signed=false;
        a.number[0]=0x43C7A7F4AC8DC7A3;
        a.number[1]=0x16CAD6F88423B164;
        a.number[2]=0x25970C2B53A8E9E9;
        a.number[3]=0x1101B4C7590887BC;
        b.size=4;
        b.is_signed=false;
        b.number[0]=0xA4867513BA6EDB7;
        b.number[0]=0xEE09B93519940300;
        b.number[0]=0x80A47E44D697FF6D;
        b.number[0]=0xE6876CD5F6FDB52;
        vec->push(a);
        vec->push(b);

        VariablesGen variablesGen;
        variablesGen.public_gen(vec);
         Vec<Any>* public_var = variablesGen.getPublic();
         printf("%d\n",public_var->getSize());
        for (int i = 0; i < public_var->getSize(); i++) {
           Any any = public_var->get(i);
           if (any.getType() == Type::_u256) {
               skm::BigInt v = any.getBigInt();
               skm::BigIntOper::print_hex(&v);
           } else if (any.getType() == Type::_bool_struct) {
               printf("%d", any.getBoolStruct().value);
           }
        }
        printf("\n");
    }
}
#endif //SNARKVM_CUDA_TEST_VARIABLES_GEN_CUH
