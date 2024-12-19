//
// Created by fil on 24-8-31.
//
#pragma once
#ifndef SNARKVM_CUDA_BIGINT_CUH
#define SNARKVM_CUDA_BIGINT_CUH
#include "basic_oper.cuh"

namespace skm {
    class BigIntOper {
    private:
        // 函数：比较两个 128 位无符号整数
        __device__ static bool compare_uint128(const BigInt* a, const BigInt* b);
        // 函数：减去两个 128 位无符号整数
        __device__ static BigInt* subtract_uint128(const BigInt* a, const BigInt* b);

    protected:
        __device__ static BigInt* zero(int length, bool is_signed);
        __device__ static BigInt* zero(int length, bool is_signed, Mode mod);
        __device__ static bool is_zero(const BigInt* a);
        __device__ static void to_zero(BigInt* a);
        __device__ static bool isNegative(const BigInt* a);
        __device__ static void setNegative(BigInt& a, bool negative);
        //将补码转为原码
        __device__ static BigInt* to_original(BigInt* a);
        __device__ static void insert(BigInt* a, int begin, int len, __uint64_t val);
        __device__ static BigInt* shiftLeft(BigInt* a, __uint32_t shift); //左移，并扩大原数
        __device__ static BigInt* mod(const BigInt* a, const BigInt* mod);
        // 大整数除法运算，获取余数用于十进制转换
        __device__ static U64 divmod_bigint(BigInt *a, U64 divisor);
        __device__ static U8 _addcarry_u64(const U8 carry_in, const U64 src1 , const U64 src2, U64 *dst);
        __device__ static U8 _subborrow_u64(const U8 borrow_in, const U64 src1, const U64 src2, U64 *dst);
        __device__ static BigInt* sub_w(const BigInt* a, const BigInt* b, BigInt* result);
    public:
        __device__ static void copy_to(const BigInt* a, BigInt* b);
        __device__ static BigInt* clone(const BigInt* a);
        //存储按照低位在前，高位在后的顺序存储,方便编程;  函数的参数则高位在前，方便记忆
        __device__ static BigInt* CreatInt128(__uint64_t num1, __uint64_t num2, bool is_signed);
        __device__ static BigInt* CreatInt128(__uint64_t num1, __uint64_t num2, bool is_signed, Mode mod);
        __device__ static BigInt* CreatInt256(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, bool is_signed);
        __device__ static BigInt* CreatInt256(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, bool is_signed, Mode mod);
        //__device__ static BigInt* CreatInt384(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, __uint64_t num5, __uint64_t num6, bool is_signed);
        //__device__ static BigInt* CreatInt384(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, __uint64_t num5, __uint64_t num6, bool is_signed, Mode mod);
        //__device__ static BigInt* CreatInt512(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, __uint64_t num5, __uint64_t num6, __uint64_t num7, __uint64_t num8, bool is_signed);
        //__device__ static BigInt* CreatInt512(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, __uint64_t num5, __uint64_t num6, __uint64_t num7, __uint64_t num8, bool is_signed, Mode mod);
        __device__ static BigInt* add_w(const BigInt* a, const BigInt* b);
        __device__ static BigInt* add_w_change(BigInt* a, const BigInt* b);
        __device__ static BigInt* add(const BigInt* a, const BigInt* b);
        __device__ static BigInt* mul_w(const BigInt* a, const BigInt* b);
        __device__ static BigInt* mul(const BigInt* a, const BigInt* b);
        __device__ static BigInt* sub_w(const BigInt* a, const BigInt* b);
        __device__ static BigInt* sub_w_change(BigInt* a, const BigInt* b);
        __device__ static BigInt* sub(const BigInt* a, const BigInt* b);
        __device__ static bool is_eq(const BigInt* a, const BigInt* b);
        __device__ static bool lt(const BigInt* a, const BigInt* b);
        __device__ static bool is_neq(const BigInt* a, const BigInt* b);
        __device__ static bool lte(const BigInt* a, const BigInt* b);
        __device__ static bool gt(const BigInt* a, const BigInt* b);
        __device__ static bool gte(const BigInt* a, const BigInt* b);
        __device__ static BigInt* xor_(const BigInt* a, const BigInt* b);
        __device__ static BigInt* and_(BigInt* a, BigInt* b);
        __device__ static BigInt* neg(const BigInt* a);
        __device__ static BigInt* shr_w(const BigInt* a, int shift);
        __device__ static BigInt* shl_w(const BigInt* a, int shift);
        __device__ static BigInt* abs_w(const BigInt* num);
        __device__ static BigInt* abs(const BigInt* a);
        __device__ static BigInt* div_w(const BigInt* dividend, const BigInt* divisor);
        __device__ static BigInt* div(const BigInt* a, const BigInt* b);
        __device__ static BigInt* ternary(bool condition, BigInt* a, BigInt* b);
        __device__ static void destory(BigInt* a);
        // 打印大整数
        __device__ static void print_bigint(const BigInt* num);
        __device__ static void print_hex(const BigInt* num);
        // 设备端模板函数，允许精度丢失的 BigInt 转普通整型（转化为64位以下的整形）
        template<typename To>
        __device__ static
        To cast_lossy(const BigInt& bigInt);
        __device__ static Vec<Bool>* to_bits_le(const BigInt* bigInt);
        __device__ static Vec<Bool>* to_bits_le(const BigInt* bigInt, Mode mode);
        __device__ static Vec<Bool>* to_bits_be(const BigInt* bigInt);
        __device__ static Vec<Bool>* to_bits_be(const BigInt* bigInt, Mode mode);
        __device__ static Vec<skm::BigInt*>* clone_vec(Vec<skm::BigInt*>* src);
        __device__ static void copy_from_slice(Vec<skm::BigInt*>* target,  Vec<skm::BigInt*> *src, u64 start, u64 len);
        __device__ static void copy_from_slice(Vec<skm::BigInt*>* target,  Vec<skm::BigInt*> *src);
        __device__ static Vec<skm::BigInt*>* clone_from_slice( Vec<skm::BigInt*>* target , Vec<skm::BigInt*>* src);
        __device__ static Vec<skm::BigInt*>* clone_from_slice(Vec<skm::BigInt*>* target, Vec<skm::BigInt*>* src , u64 start, u64 len);
    };
}



#endif //SNARKVM_CUDA_BIGINT_CUH
