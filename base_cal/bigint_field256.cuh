//
// Created by fil on 24-8-31.
//
#pragma once
#ifndef SNARKVM_CUDA_BIGINT_FIELD256_CUH
#define SNARKVM_CUDA_BIGINT_FIELD256_CUH
#include "bigint.cuh"
namespace skm {
    __device__ __constant__  extern U64 INV;
    __device__  __constant__ extern BigInt MAX_256_MODULUS;
    __device__  __constant__ extern BigInt R2;
    __device__  __constant__ extern BigInt ONE;
    __device__  __constant__ extern BigInt ZERO;
    __device__ __constant__  extern int BASE_FIELD_SIZE_DATA;

    __device__ __constant__ extern int FIELD_HASH_HEAD_SIZE;
    __device__  __constant__ extern u64 FIELD_SIZE;          //一个field的长度
    __device__ __constant__  extern int u64_limbs;
    __device__  __constant__ extern int REPR_SHAVE_BITS;
    __device__ __constant__ extern int SERIALIZED_SIZE;
    __device__ __constant__ extern int BIT_SIZE;

    __device__  extern Vec<Bool> *FIELD_HASH_HEAD;        //field的头,使用时初始化


    class BigIntOperField256 : public BigIntOper {
    private:
        __device__ static u64 mac(u64 a, u64 b, u64 c, u64 *carry);
        __device__ static void mac_discard(u64 a, u64 b, u64 c, u64 *carry);
        __device__ static u64 mac_with_carry(u64 a, u64 b, u64 c, u64 *carry);
        __device__ static bool is_even(const BigInt* a);
        __device__ static bool is_odd(const BigInt* a);
        __device__ static void div2(BigInt *a);
        __device__ static void divn(BigInt *a, int n);
        __device__ static bool add_nocarry(BigInt *a, const BigInt *b);
        __device__ static bool sub_noborrow(BigInt *a, const BigInt *b);
        __device__ static u64 adc(u64* a, u64 b, u64 carry);
        __device__ static void mont_reduce(BigInt* self, const u64 r0, u64* r1, u64* r2, u64* r3, u64* r4, u64* r5, u64* r6, u64* r7);
        __device__ static BigInt* square_in_place(BigInt* self);
        __device__ static void sub_assign(BigInt *a, const BigInt *b);
        __device__ static void reduce(BigInt *a);

//        __device__ static Vec<Vec<Bool>*>* chunks(Vec<Bool> *vec, U32 chunk_size) {
//            const int numChunks = (vec->getSize() + chunk_size - 1) / chunk_size; // 计算块的数量
//            Vec<Vec<Bool>*> *newVecs = Vec<Vec<Bool>*>::init(numChunks);
//            Vec<Bool>* unit = nullptr;
//            for (int i = 0; i < vec->getSize(); i++) {
//               if (i % chunk_size == 0) {
//                   unit = Vec<Bool>::init(chunk_size);
//                   newVecs->push(unit);
//               }
//                unit->push(vec->get(i));
//            }
//            return newVecs;
//        }

    public:
        __device__ static void add_head(Vec<Bool> *vec);
        __device__ static BigInt* inverse(const BigInt *a);
        __device__ static BigInt* add(const BigInt *a, const BigInt *b);
        __device__ static BigInt* add_change(BigInt *a, const BigInt *b);
        __device__ static BigInt* mul(const BigInt *a, const BigInt *b);
        __device__ static BigInt* mul_change(BigInt *a, const BigInt *b);
        __device__ static BigInt*  div(const BigInt *a, const BigInt *b);
        __device__ static BigInt* sub(const BigInt *a, const BigInt *b);
        __device__ static BigInt* sub_change(BigInt *a, const BigInt *b);
        // 设备端模板函数，允许精度丢失的 BigInt 丢失精度
        template<typename To>
        __device__ static
        To* cast_lossy(const BigInt *a);
        //转为正常big_int数值
        __device__ static BigInt* to_normal_big_int(const BigInt *a);
        template<typename T>
        __device__ static BigInt* from(T a);
        //正常big_int转为Field256数值
        __device__ static BigInt* to_filed256_big_int(const BigInt *a);
        __device__ static BigInt* from_bigint(const BigInt *a);
        __device__ static bool is_valid(const BigInt *a);
        __device__ static Vec<Bool>* to_bits_le(const BigInt* bigInt);
        __device__ static Vec<Bool>* to_bits_be(const BigInt* bigInt);
        __device__ static Vec<Bool>* to_bits_le(const BigInt* bigInt, Mode mode);
        __device__ static Vec<Bool>* to_bits_be(const BigInt* bigInt, Mode mode);
        __device__ static Vec<BigInt*>* to_fields(Vec<Bool> *vec_src, int chunk_size);
        __device__ static BigInt* from_bits_le(Vec<Bool>* vec);
        __device__ static BigInt* from_bits_le(Vec<bool>* vec);
        /* todo from_bits_le的另外一种实现
        __device__ static BigInt* from_bits_le(Vec<bool>* vec) {
            int num_bits = vec->getSize();
            BigInt* output = skm::BigIntOperField256::clone(&ZERO);
            BigInt* coefficient = skm::BigIntOperField256::clone(&ONE);
            for (int i = 0; i < num_bits; i++) {
                if (vec->get(i)) {
                    output = add_change(output, coefficient);
                }
                coefficient = double_(coefficient);
            }
            destory(coefficient);
            return output;
        }
         */
        __device__ static BigInt* double_(BigInt* a);
        __device__ static BigInt* pow(const BigInt* a, const BigInt* b);
        __device__ static BigInt* square(const BigInt* self);
        __device__ static BigInt* square_change(BigInt* self);
        __device__ static BigInt* from_random_bytes(Vec<u8>* bytes);
        __device__ static BigInt* to_bigint(const BigInt* self);
        __device__ static BigInt* from_bytes_be_mod_order(Vec<u8>* bytes);
        __device__ static BigInt* from_bytes_le_mod_order(Vec<u8>* bytes);
        __device__ static BigInt* new_domain_separator(char* str);
    };

    //================================实现==========================================
    template<typename T>
    __device__  BigInt* BigIntOperField256::from(T a) {
        BigInt *rst = zero(4, false);
        rst->number[0] = (U64) a;
    }

    // 设备端模板函数，允许精度丢失的 BigInt 丢失精度
    template<typename To>
    __device__ To* BigIntOperField256::cast_lossy(const BigInt *a) {
        int size = sizeof(To);
        if (size <= 8) { //64位以下的处理
            return static_cast<To>(a->number[0]);
        }

        //256位转化为128位
        if (std::is_same<To, U128>::value) {
            BigInt *result = zero(2, false);
            result->number[0] = a->number[0];
            result->number[1] = a->number[1];
            return result;
        }
        if (std::is_same<To, I128>::value) {
            BigInt *result = zero(2, true);
            result->number[0] = a->number[0];
            result->number[1] = a->number[1];
            return result;
        }
    }
}
#endif //SNARKVM_CUDA_BIGINT_FIELD256_CUH
