//
// Created by fil on 24-10-16.
//
#include "bigint_field256.cuh"
#include "cache/VmCache.cuh"
#include "cache/VmCacheVecOper.h"

namespace skm {
        __device__ __constant__  U64 INV = 725501752471715839ULL;
        __device__  __constant__  BigInt MAX_256_MODULUS = BigInt{4, false, {(U64)725501752471715841ULL, (U64)6461107452199829505ULL, (U64)6968279316240510977ULL,(U64)1345280370688173398ULL}};
        __device__  __constant__  BigInt R2 = BigInt{4, false, {(U64)2726216793283724667ULL, (U64)14712177743343147295ULL, (U64)12091039717619697043ULL,(U64)81024008013859129ULL}};
        __device__  __constant__  BigInt ONE  = BigInt{4, false, {(U64)9015221291577245683ULL, (U64)8239323489949974514ULL, (U64)1646089257421115374ULL,(U64)958099254763297437ULL}};
        __device__  __constant__  BigInt ZERO  = BigInt{4, false, {0, 0, 0,0}};
        __device__ __constant__  int BASE_FIELD_SIZE_DATA = 252;

        __device__ __constant__ int FIELD_HASH_HEAD_SIZE = 26;
        __device__  __constant__ u64 FIELD_SIZE = 253;          //一个field的长度
        __device__ __constant__  int u64_limbs = 4;
        __device__  __constant__ int REPR_SHAVE_BITS = 3;
        __device__ __constant__ int SERIALIZED_SIZE = 32;
        __device__ __constant__ int BIT_SIZE = 0;

        __device__ Vec<Bool> *FIELD_HASH_HEAD = nullptr;        //field的头,使用时初始化
        __device__ int field_is_init = 0;

        __device__  u64 BigIntOperField256::mac(u64 a, u64 b, u64 c, u64 *carry) {
            __uint128_t tmp = (__uint128_t) a + (__uint128_t) b * (__uint128_t) c;
            *carry = u64(tmp >> 64);
            return (u64) tmp;
        }

        __device__ void BigIntOperField256::mac_discard(u64 a, u64 b, u64 c, u64 *carry) {
            __uint128_t tmp = (__uint128_t) a + (__uint128_t) b * (__uint128_t) c;
            *carry = u64(tmp >> 64);
        }

        __device__  u64 BigIntOperField256::mac_with_carry(u64 a, u64 b, u64 c, u64 *carry) {
            __uint128_t tmp = (__uint128_t) a + (__uint128_t) b * (__uint128_t) c + (__uint128_t) (*carry);
            *carry = u64(tmp >> 64);
            return (u64)tmp;
        }

        __device__  bool BigIntOperField256::is_even(const BigInt* a) {
            return !is_odd(a);
        }

        __device__  bool BigIntOperField256::is_odd(const BigInt* a) {
            return (a->number[0] & 1) == 1;
        }

        __device__  void BigIntOperField256::div2(BigInt *a) {
            u64 t = 0;
            for (int i = a->size - 1; i >= 0; --i) {
                u64 t2 = a->number[i] << 63;
                a->number[i] >>= 1;
                a->number[i] |= t;
                t = t2;
            }
        }

        __device__  void BigIntOperField256::divn(BigInt *a, int n) {
            if (n >= 64 * 4) {
                to_zero(a);
                return;
            }
            while (n >= 64) {
                u64 t = 0;
                for (int i = 0; i < a->size; ++i) {
                    skm::BasicOper::swap(&t, &(a->number[i]));
                }
                n -= 64;
            }

            if (n > 0) {
                u64 t = 0;
                for (int i = 0; i < a->size; i++) {
                    u64 t2 = a->number[i] << (64 - n);
                    a->number[i] >>= n;
                    a->number[i] |= t;
                    t = t2;
                }
            }
        }

        __device__  bool BigIntOperField256::add_nocarry(BigInt *a, const BigInt *b) {
            U8 carry = 0;
            carry = _addcarry_u64(carry, a->number[0], b->number[0], &a->number[0]);
            carry = _addcarry_u64(carry, a->number[1], b->number[1], &a->number[1]);
            carry = _addcarry_u64(carry, a->number[2], b->number[2], &a->number[2]);
            carry = _addcarry_u64(carry, a->number[3], b->number[3], &a->number[3]);
            return carry != 0;
        }

        __device__  bool BigIntOperField256::sub_noborrow(BigInt *a, const BigInt *b) {
            U8 borrow = 0;
            borrow = _subborrow_u64(borrow, a->number[0], b->number[0], &a->number[0]);
            borrow = _subborrow_u64(borrow, a->number[1], b->number[1], &a->number[1]);
            borrow = _subborrow_u64(borrow, a->number[2], b->number[2], &a->number[2]);
            borrow = _subborrow_u64(borrow, a->number[3], b->number[3], &a->number[3]);
            return borrow != 0;
        }

        __device__  u64 BigIntOperField256::adc(u64* a, u64 b, u64 carry) {
            __uint128_t tmp = __uint128_t(*a) + __uint128_t(b) + __uint128_t(carry);
            *a = (u64)tmp;
            return (u64)(tmp >> 64);
        }

        __device__  void BigIntOperField256::mont_reduce(BigInt* self, const u64 r0, u64* r1, u64* r2, u64* r3, u64* r4, u64* r5, u64* r6, u64* r7) {
            u64 k = skm::BasicOper::mul_w(r0, INV);
            u64 carry = 0;
            mac_with_carry(r0, k, MAX_256_MODULUS.number[0], &carry);
            *r1 = mac_with_carry(*r1, k, MAX_256_MODULUS.number[1], &carry);
            *r2 = mac_with_carry(*r2, k, MAX_256_MODULUS.number[2], &carry);
            *r3 = mac_with_carry(*r3, k, MAX_256_MODULUS.number[3], &carry);
            carry = adc(r4, 0, carry);
            u64 carry2 = carry;
            k = skm::BasicOper::mul_w(*r1, INV);
            carry = 0;
            mac_with_carry(*r1, k, MAX_256_MODULUS.number[0], &carry);
            *r2 = mac_with_carry(*r2, k, MAX_256_MODULUS.number[1], &carry);
            *r3 = mac_with_carry(*r3, k, MAX_256_MODULUS.number[2], &carry);
            *r4 = mac_with_carry(*r4, k, MAX_256_MODULUS.number[3], &carry);
            carry = adc(r5, carry2, carry);
            carry2 = carry;
            k = skm::BasicOper::mul_w(*r2, INV);
            carry = 0;
            mac_with_carry(*r2, k,MAX_256_MODULUS.number[0], &carry);
            *r3 = mac_with_carry(*r3, k, MAX_256_MODULUS.number[1], &carry);
            *r4 = mac_with_carry(*r4, k, MAX_256_MODULUS.number[2], &carry);
            *r5 = mac_with_carry(*r5, k, MAX_256_MODULUS.number[3], &carry);
            carry = adc(r6, carry2, carry);
            carry2 = carry;
            k = skm::BasicOper::mul_w(*r3, INV);
            carry = 0;
            mac_with_carry(*r3, k, MAX_256_MODULUS.number[0], &carry);
            *r4 = mac_with_carry(*r4, k, MAX_256_MODULUS.number[1], &carry);
            *r5 = mac_with_carry(*r5, k, MAX_256_MODULUS.number[2], &carry);
            *r6 = mac_with_carry(*r6, k, MAX_256_MODULUS.number[3], &carry);
            adc(r7, carry2, carry);
            self->number[0] = *r4;
            self->number[1] = *r5;
            self->number[2] = *r6;
            self->number[3] = *r7;
            reduce(self);
        }

        __device__  BigInt* BigIntOperField256::square_in_place(BigInt* self) {
            u64 carry = 0;
            u64 r1 = mac_with_carry(0, self->number[0], self->number[1], &carry);
            u64 r2 = mac_with_carry(0, self->number[0], self->number[2], &carry);
            u64 r3 = mac_with_carry(0, self->number[0], self->number[3], &carry);
            u64 r4 = carry;
            carry = 0;
            r3 = mac_with_carry(r3, self->number[1], self->number[2], &carry);
            r4 = mac_with_carry(r4, self->number[1], self->number[3], &carry);
            u64 r5 = carry;
            carry = 0;
            r5 = mac_with_carry(r5, self->number[2], self->number[3], &carry);
            u64 r6 = carry;

            u64 r7 = r6 >> 63;
            r6 = (r6 << 1) | (r5 >> 63);
            r5 = (r5 << 1) | (r4 >> 63);
            r4 = (r4 << 1) | (r3 >> 63);
            r3 = (r3 << 1) | (r2 >> 63);
            r2 = (r2 << 1) | (r1 >> 63);
            r1 = r1 << 1;

            carry = 0;
            u64 r0 = mac_with_carry(0, self->number[0], self->number[0], &carry);
            carry = adc(&r1, 0, carry);
            r2 = mac_with_carry(r2, self->number[1], self->number[1], &carry);
            carry = adc(&r3, 0, carry);
            r4 = mac_with_carry(r4, self->number[2], self->number[2], &carry);
            carry = adc(&r5, 0, carry);
            r6 = mac_with_carry(r6, self->number[3], self->number[3], &carry);
            adc(&r7, 0, carry);

            mont_reduce(self, r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            return self;
        }

        __device__  void BigIntOperField256::sub_assign(BigInt *a, const BigInt *b) {
            if (gt(b, a)) {
                add_nocarry(a, &MAX_256_MODULUS);
            }
            sub_noborrow(a, b);
        }

        __device__  void BigIntOperField256::reduce(BigInt *a) {
            if (!is_valid(a)) {
                sub_noborrow(a, &MAX_256_MODULUS);
            }
        }


        __device__ void BigIntOperField256::add_head(Vec<Bool> *vec) {
            if (nullptr == FIELD_HASH_HEAD) { // 初始化头
                printf("初始化field头开始!%d");
                skm::acquire_lock(&field_is_init);
                if (nullptr == FIELD_HASH_HEAD) {
                    const int size = 26;
                    bool HEAD_ARRAY[size] = {false, false, false, true, false, false, false, false, false, false, true,
                                             false,
                                             true, true, true, true, true, true, false, false, false, false, false,
                                             false,
                                             false, false};
                    Vec<Bool> *headVec = VmCache::getBoolStructLst(size);
                    for (int i = 0; i < size; i++) {
                        headVec->push(Bool{HEAD_ARRAY[i], Mode::Constant});
                    }

                    FIELD_HASH_HEAD = headVec;
                }
                printf("初始化field头完毕!%d");
                skm::release_lock(&field_is_init);
            }

            //添加头
            vec->insert(0, FIELD_HASH_HEAD);
        }
        __device__  BigInt* BigIntOperField256::inverse(const BigInt *a) {
            BigInt *rst = zero(a->size, a->is_signed);
            if (is_zero(a)) {
                return rst;
            }
            BigInt *one = CreatInt256(1, 0, 0, 0, false);
            BigInt *u = clone(a);
            BigInt *v = clone(&MAX_256_MODULUS);
            BigInt *b = clone(&R2);
            BigInt *c = zero(a->size, a->is_signed);

            while (is_neq(u, one) && is_neq(v, one)) {
                while (is_even(u)) {
                    div2(u);
                    if (is_even(b)) {
                        div2(b);
                    } else {
                        add_nocarry(b, &MAX_256_MODULUS);
                        div2(b);
                    }
                }

                while (is_even(v)) {
                    div2(v);
                    if (is_even(c)) {
                        div2(c);
                    } else {
                        add_nocarry(c, &MAX_256_MODULUS);
                        div2(c);
                    }
                }

                if (lt(v, u)) {
                    sub_noborrow(u, v);
                    sub_assign(b, c);
                } else {
                    sub_noborrow(v, u);
                    sub_assign(c, b);
                }
            }

            destory(rst);
            destory(v);
            destory(one);
            if (is_eq(u, one)) {
                destory(u);
                destory(c);
                return b;
            } else {
                destory(u);
                destory(b);
                return c;
            }
        }

        __device__  BigInt* BigIntOperField256::add(const BigInt *a, const BigInt *b) {
            BigInt *result = clone(a);
            return add_change(result, b);
        }

        __device__  BigInt* BigIntOperField256::add_change(BigInt *a, const BigInt *b) {
            add_nocarry(a, b);
            reduce(a);
            return a;
        }

        __device__  BigInt* BigIntOperField256::mul(const BigInt *a, const BigInt *b) {
            BigInt *r = zero(b->size, b->is_signed);
            u64 carry1 = 0;
            u64 carry2 = 0;
            const BigInt MODULUS = MAX_256_MODULUS;
            for (int j = 0; j < b->size; ++j) {
                r->number[0] = mac(r->number[0], a->number[0], b->number[j], &carry1);
                u64 k = skm::BasicOper::mul_w(r->number[0], INV);
                mac_discard(r->number[0], k, MODULUS.number[0], &carry2);
                r->number[1] = mac_with_carry(r->number[1], a->number[1], b->number[j], &carry1);
                r->number[0] = mac_with_carry(r->number[1], k, MODULUS.number[1], &carry2);

                r->number[2] = mac_with_carry(r->number[2], a->number[2], b->number[j], &carry1);
                r->number[1] = mac_with_carry(r->number[2], k, MODULUS.number[2], &carry2);

                r->number[3] = mac_with_carry(r->number[3], a->number[3], b->number[j], &carry1);
                r->number[2] = mac_with_carry(r->number[3], k, MODULUS.number[3], &carry2);
                r->number[3] = carry1 + carry2;
            }
            if (gte(r, &MODULUS)) {
                BigInt *rst = sub_w(r, &MODULUS);
                destory(r);
                return rst;
            }
            r->mod = a->mod;
            return r;
        }

        __device__  BigInt* BigIntOperField256::mul_change(BigInt *a, const BigInt *b) {
            BigInt *rst = mul(a, b);
            copy_to(rst, a);
            destory(rst);
            return a;
        }

        __device__  BigInt* BigIntOperField256:: div(const BigInt *a, const BigInt *b) {
            BigInt *t = inverse(b);
            BigInt *r = mul(a, t);
            destory(t);
            return r;
        }

        __device__  BigInt* BigIntOperField256::sub(const BigInt *a, const BigInt *b) {
            BigInt *rst = clone(a);
            return sub_change(rst, b);
        }

        __device__  BigInt* BigIntOperField256::sub_change(BigInt *a, const BigInt *b) {
            if (gt(b, a)) {
                add_nocarry(a, &MAX_256_MODULUS);
            }
            sub_noborrow(a, b);
            return a;
        }

//转为正常big_int数值
        __device__  BigInt* BigIntOperField256::to_normal_big_int(const BigInt *a) {
            BigInt *tmp = clone(a);
            U64 k = skm::BasicOper::mul_w(tmp->number[0], INV);
            U64 carry = 0;
            mac_with_carry(tmp->number[0], k, MAX_256_MODULUS.number[0], &carry);
            tmp->number[1] = mac_with_carry(tmp->number[1], k, MAX_256_MODULUS.number[1], &carry);
            tmp->number[2] = mac_with_carry(tmp->number[2], k, MAX_256_MODULUS.number[2], &carry);
            tmp->number[3] = mac_with_carry(tmp->number[3], k, MAX_256_MODULUS.number[3], &carry);
            tmp->number[0] = carry;
            k = skm::BasicOper::mul_w(tmp->number[1], INV);
            carry = 0;
            mac_with_carry(tmp->number[1], k, MAX_256_MODULUS.number[0], &carry);
            tmp->number[2] = mac_with_carry(tmp->number[2], k, MAX_256_MODULUS.number[1], &carry);
            tmp->number[3] = mac_with_carry(tmp->number[3], k, MAX_256_MODULUS.number[2], &carry);
            tmp->number[0] = mac_with_carry(tmp->number[0], k, MAX_256_MODULUS.number[3], &carry);
            tmp->number[1] = carry;

            k = skm::BasicOper::mul_w(tmp->number[2], INV);
            carry = 0;
            mac_with_carry(tmp->number[2], k, MAX_256_MODULUS.number[0], &carry);
            tmp->number[3] = mac_with_carry(tmp->number[3], k, MAX_256_MODULUS.number[1], &carry);
            tmp->number[0] = mac_with_carry(tmp->number[0], k, MAX_256_MODULUS.number[2], &carry);
            tmp->number[1] = mac_with_carry(tmp->number[1], k, MAX_256_MODULUS.number[3], &carry);
            tmp->number[2] = carry;

            k = skm::BasicOper::mul_w(tmp->number[3], INV);
            carry = 0;
            mac_with_carry(tmp->number[3], k, MAX_256_MODULUS.number[0], &carry);
            tmp->number[0] = mac_with_carry(tmp->number[0], k, MAX_256_MODULUS.number[1], &carry);
            tmp->number[1] = mac_with_carry(tmp->number[1], k, MAX_256_MODULUS.number[2], &carry);
            tmp->number[2] = mac_with_carry(tmp->number[2], k, MAX_256_MODULUS.number[3], &carry);
            tmp->number[3] = carry;

            return tmp;
        }

//正常big_int转为Field256数值
        __device__  BigInt* BigIntOperField256::to_filed256_big_int(const BigInt *a) {
            if (is_zero(a)) {
                return zero(4, false, a->mod);
            } else if (is_valid(a)) {
                return mul(a, &R2);
            }
            return nullptr;
        }

        __device__  BigInt* BigIntOperField256::from_bigint(const BigInt *a) {
            return to_filed256_big_int(a);
        }

        __device__  bool BigIntOperField256::is_valid(const BigInt *a) {
            return lt(a, &MAX_256_MODULUS);
        }

        __device__  Vec<Bool>* BigIntOperField256::to_bits_le(const BigInt* bigInt) {
            return to_bits_le(bigInt, Mode::Private);
        }

        __device__  Vec<Bool>* BigIntOperField256::to_bits_be(const BigInt* bigInt) {
            return to_bits_be(bigInt, Mode::Private);
        }

        __device__  Vec<Bool>* BigIntOperField256::to_bits_le(const BigInt* bigInt, Mode mode) {
            BigInt *newVal = to_normal_big_int(bigInt);
            Vec < Bool > *bitle = BigIntOper::to_bits_le(newVal, mode);
            destory(newVal);
            bitle->pop();
            bitle->pop();
            bitle->pop();
            return bitle;
        }

        __device__  Vec<Bool>* BigIntOperField256::to_bits_be(const BigInt* bigInt, Mode mode) {
            Vec < Bool > *vec = to_bits_le(bigInt, mode);
            return vec->reverse();
        }

        __device__  Vec<BigInt*>* BigIntOperField256::to_fields(Vec<Bool> *vec_src, int chunk_size) {
            Vec < Bool > *vec = VmCacheVecOper::cloneBoolStruct(vec_src, vec_src->getSize() + 32);
            add_head(vec);
            vec->push(Bool
            { true, Mode::Constant });
            Vec < Vec < Bool > * > *vecChunks = VmCacheVecOper::chunksBoolStruct(vec, chunk_size);
            Vec < BigInt * > *list = VmCache::getBigIntLst(vecChunks->getSize());
            //处理field
            for (int i = 0; i < vecChunks->getSize(); i++) {
                list->push(from_bits_le(vecChunks->get(i)));
            }
            return list;
        }

        __device__  BigInt* BigIntOperField256::from_bits_le(Vec<Bool>* vec) {
            Vec < bool > *new_vec = VmCache::getBoolLst(vec->getSize());
            for (int i = 0; i < vec->getSize(); i++) {
                new_vec->push(vec->get(i).value);
            }
            return from_bits_le(new_vec);
        }

        __device__  BigInt* BigIntOperField256::from_bits_le(Vec<bool>* vec) {
            Vec < Vec < bool > * > *vecs = VmCacheVecOper::chunksBool(vec, 64);
            BigInt *res = zero(4, false);
            for (int i = 0; i < vecs->getSize(); i++) {
                u64 acc = 0;
                Vec < bool > *t = vecs->get(i);
                for (int j = t->getSize() - 1; j >= 0; j--) {
                    acc <<= 1;
                    acc += t->get(j);
                }
                res->number[i] = acc;
            }
            VmCache::returnBoolLstLst(vecs);
            return res;
        }
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

        __device__  BigInt* BigIntOperField256::double_(BigInt* a) {
            return add_change(a, a);
        }

        __device__  BigInt* BigIntOperField256::pow(const BigInt* a, const BigInt* b) {
            BigInt *output = clone(&ONE);
            if (b->mod == Mode::Constant) {
                Vec < Bool > *vecs = BigIntOperField256::to_bits_be(b);
                for (int i = 0; i < vecs->getSize(); i++) {
                    square_change(output);
                    if (vecs->get(i).value) {
                        mul_change(output, a);
                    }
                }
                VmCache::returnBoolStructLst(vecs);
            } else {
                Vec < Bool > *vecs = BigIntOperField256::to_bits_be(b);
                for (int i = 0; i < vecs->getSize(); i++) {
                    // Square the output.
                    square_change(output);
                    // If `bit` is `true, set the output to `output * self`.
                    BigInt *tmp = output;
                    BigInt *mul_tmp = mul(output, a);
                    output = skm::BasicOper::ternary(vecs->get(i).value, mul_tmp, tmp);
                    if (output != mul_tmp) {
                        destory(mul_tmp);
                    }
                    if (output != tmp) {
                        destory(tmp);
                    }
                }
                VmCache::returnBoolStructLst(vecs);
            }
            return output;
        }

        __device__  BigInt* BigIntOperField256::square(const BigInt* self) {
            BigInt *rst = clone(self);
            return square_change(rst);
        }
        __device__  BigInt* BigIntOperField256::square_change(BigInt* self) {
            BigInt *temp = self;
            square_in_place(temp);
            return temp;
        }

        __device__  BigInt* BigIntOperField256::from_random_bytes(Vec<u8>* bytes) {
            int size = u64_limbs * 8 + 1;
            Vec < u8 > *result_bytes = VmCache::getU8Lst(size);
            result_bytes->setSize(size);
            for (int i = 0; i < result_bytes->getSize(), i < bytes->getSize(); i++) {
                result_bytes->set(i, bytes->get(i));
            }
            Vec < u8 > *last_limb_mask = skm::BasicOper::to_le_bytes(VmCache::getU8Lst(8), MAX_U64 >> REPR_SHAVE_BITS);
            Vec < u8 > *last_bytes_mask = VmCache::getU8Lst(9);
            last_bytes_mask->setSize(9);
            last_bytes_mask->copy_from_slice(last_limb_mask, 0, 8);
            int output_byte_size = SERIALIZED_SIZE;
            int flag_location = output_byte_size - 1;
            int flag_location_in_last_limb = flag_location - (8 * (u64_limbs - 1));
            Vec < u8 > *last_bytes = VmCacheVecOper::sub_array_change_u8(result_bytes, 8 * (u64_limbs - 1));
            u8 flags_mask = skm::BasicOper::checked_shl(MAX_U8, 8 - BIT_SIZE);
            u8 flags = 0;
            for (int i = 0; i < last_bytes->getSize(), i < last_bytes_mask->getSize(); i++) {
                if (i == flag_location_in_last_limb) {
                    flags = last_bytes->get(i) & flags_mask;
                }
                last_bytes->set(i, last_bytes->get(i) & last_bytes_mask->get(i));
            }
            Vec < u64 > *u64Vec = VmCache::getU64Lst(4);
            u64Vec->setSize(4);
            u64 tmp_number = 0;
            for (int i = 0, index = 0, mod = 0; i < u64_limbs * 8; i++) {
                mod = i % 8;
                tmp_number |= (u64) (result_bytes->get(i)) << (mod * 8);
                if (mod == 7) {
                    u64Vec->set(index, tmp_number);
                    tmp_number = 0;
                    index++;
                }
            }
            BigInt *rst_pre = BigIntOperField256::CreatInt256(u64Vec->get(0), u64Vec->get(1), u64Vec->get(2),
                                                              u64Vec->get(3), false);
            BigInt *rst = from_bigint(rst_pre);
            destory(rst_pre);
            VmCache::returnU8EmptyLstOnly(last_bytes);
            VmCache::returnU8Lst(result_bytes);
            VmCache::returnU8Lst(last_limb_mask);
            VmCache::returnU8Lst(last_bytes_mask);
            VmCache::returnU64Lst(u64Vec);
            return rst;
        }

        __device__  BigInt* BigIntOperField256::to_bigint(const BigInt* self) {
            u64 r[4];
            for (int i = 0; i < 4; i++) { r[i] = self->number[i]; }
            // Montgomery Reduction
            u64 k = skm::BasicOper::mul_w(r[0], INV);
            u64 carry = 0;
            mac_with_carry(r[0], k, MAX_256_MODULUS.number[0], &carry);
            r[1] = mac_with_carry(r[1], k, MAX_256_MODULUS.number[1], &carry);
            r[2] = mac_with_carry(r[2], k, MAX_256_MODULUS.number[2], &carry);
            r[3] = mac_with_carry(r[3], k, MAX_256_MODULUS.number[3], &carry);
            r[0] = carry;

            k = skm::BasicOper::mul_w(r[1], INV);
            carry = 0;
            mac_with_carry(r[1], k, MAX_256_MODULUS.number[0], &carry);
            r[2] = mac_with_carry(r[2], k, MAX_256_MODULUS.number[1], &carry);
            r[3] = mac_with_carry(r[3], k, MAX_256_MODULUS.number[2], &carry);
            r[0] = mac_with_carry(r[0], k, MAX_256_MODULUS.number[3], &carry);
            r[1] = carry;

            k = skm::BasicOper::mul_w(r[2], INV);
            carry = 0;
            mac_with_carry(r[2], k, MAX_256_MODULUS.number[0], &carry);
            r[3] = mac_with_carry(r[3], k, MAX_256_MODULUS.number[1], &carry);
            r[0] = mac_with_carry(r[0], k, MAX_256_MODULUS.number[2], &carry);
            r[1] = mac_with_carry(r[1], k, MAX_256_MODULUS.number[3], &carry);
            r[2] = carry;

            k = skm::BasicOper::mul_w(r[3], INV);
            carry = 0;
            mac_with_carry(r[3], k, MAX_256_MODULUS.number[0], &carry);
            r[0] = mac_with_carry(r[0], k, MAX_256_MODULUS.number[1], &carry);
            r[1] = mac_with_carry(r[1], k, MAX_256_MODULUS.number[2], &carry);
            r[2] = mac_with_carry(r[2], k, MAX_256_MODULUS.number[3], &carry);
            r[3] = carry;
            BigInt *rst = zero(4, false);
            for (int i = 0; i < 4; i++) { rst->number[i] = r[i]; }
            return rst;
        }

        __device__  BigInt* BigIntOperField256::from_bytes_be_mod_order(Vec<u8>* bytes) {
            u64 num_modulus_bytes = (u64) ((FIELD_SIZE + 7) / 8);
            u64 num_bytes_to_directly_convert = skm::BasicOper::min(num_modulus_bytes - 1, (u64) bytes->getSize());
            Vec < Vec < u8 > * > *vecs = VmCacheVecOper::split_at_u8(bytes, num_bytes_to_directly_convert);
            Vec < u8 > *leading_bytes = vecs->get(0);
            Vec < u8 > *remaining_bytes = vecs->get(1);
            Vec < u8 > *bytes_to_directly_convert = leading_bytes->reverse();
            BigInt *res = from_random_bytes(bytes_to_directly_convert);
            BigInt *window_size_pre = skm::BigIntOperField256::CreatInt256((u64) 256, 0, 0, 0, false);
            BigInt *byte_pre = skm::BigIntOperField256::CreatInt256(0, 0, 0, 0, false);

            BigInt *window_size = from_bigint(window_size_pre);
            for (int i = 0; i < remaining_bytes->getSize(); i++) {
                byte_pre->number[0] = remaining_bytes->get(i);
                BigInt *byte = from_bigint(byte_pre);
                mul_change(res, window_size);
                add_change(res, byte);
                destory(byte);
            }
            destory(byte_pre);
            destory(window_size_pre);
            destory(window_size);
            VmCache::returnU8LstLst(vecs);
            return res;
        }

        __device__  BigInt* BigIntOperField256::from_bytes_le_mod_order(Vec<u8>* bytes) {
            Vec < u8 > *bytes_copy = VmCacheVecOper::cloneU8(bytes);
            bytes_copy->reverse();
            BigInt *rst = from_bytes_be_mod_order(bytes_copy);
            VmCache::returnU8Lst(bytes_copy);
            return rst;
        }

        __device__  BigInt* BigIntOperField256::new_domain_separator(char* str) {
            Vec < u8 > *vec = skm::BasicOper::as_bytes(str);
            skm::BigInt * rst = skm::BigIntOperField256::from_bytes_le_mod_order(vec);
            VmCache::returnU8Lst(vec);
            return rst;
        }
}