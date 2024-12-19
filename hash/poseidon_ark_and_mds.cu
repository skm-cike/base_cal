//
// Created by fil on 24-10-16.
//
#include "poseidon_ark_and_mds.cuh"
#include "stdio.h"
#include "../base_cal/cache/VmCache.cuh"
#include "../base_cal/cache/VmCacheVecOper.h"
namespace ark_and_mds_pos {
    __device__ bool PoseidonGrainLFSR::next_bit() {
        bool next_bit = state[(head+62)%80]
                        ^ state[(head+51)%80]
                        ^ state[(head+38)%80]
                        ^ state[(head+23)%80]
                        ^ state[(head+13)%80]
                        ^ state[head];
        state[head]=next_bit;
        head += 1;
        head %= 80;
        return next_bit;
    }

    __device__  PoseidonGrainLFSR* PoseidonGrainLFSR::new_(bool is_sbox_an_inverse, u64 field_size_in_bits, u64 state_len, u64 num_full_rounds, u64 num_partial_rounds) {
        PoseidonGrainLFSR*  rst = new PoseidonGrainLFSR();
        bool* state = rst->state;
        state[1]=true;
        state[5]=is_sbox_an_inverse;
        u64 cur = field_size_in_bits;
        for (int i = 17; i >= 6; i--) {
            state[i] = (cur & 1) == 1;
            cur >>= 1;
        }
        cur = state_len;
        for (int i = 29; i >= 18; i--) {
            state[i] = (cur & 1) == 1;
            cur >>= 1;
        }
        cur = num_full_rounds;
        for (int i = 39; i >= 30; i--) {
            state[i] = (cur & 1) == 1;
            cur >>= 1;
        }
        cur = num_partial_rounds;
        for (int i = 49; i >= 40; i--) {
            state[i] = (cur & 1) == 1;
            cur >>= 1;
        }

        for (int i = 50; i <= 79; i++) {
            state[i]=true;
        }
        rst->field_size_in_bits=field_size_in_bits;
        rst->head=0;

        for (int i = 0; i < 160; i++) {
            rst->next_bit();
        }
        return rst;
    }
    __device__ Vec<bool>* PoseidonGrainLFSR::get_bits(Vec<bool> *bits, u64 field_size_in_bits) {
        bits->reset_size();
        for (int j = 0; j < field_size_in_bits; j++) {
            bits->push(this->get_bit());
        }
        return bits;
    }
    __device__ bool PoseidonGrainLFSR::get_bit() {
        bool new_bit = this->next_bit();
        while (!new_bit) {
            this->next_bit();
            new_bit = this->next_bit();
        }
        return this->next_bit();
    }
    __device__ Vec<skm::BigInt*>* PoseidonGrainLFSR::get_field_elements_rejection_sampling(u64 num_elements) {
        Vec<skm::BigInt*>* output = VmCache::getBigIntLst(num_elements);
        Vec<bool> *bits =  VmCache::getBoolLst(this->field_size_in_bits);
        for (int i = 0; i < num_elements; i++) {
            while (true) {
                bits = this->get_bits(bits, this->field_size_in_bits);
                bits->reverse();
                skm::BigInt *val = skm::BigIntOperField256::from_bits_le(bits);
                skm::BigInt *tmp = val;
                val = skm::BigIntOperField256::to_filed256_big_int(val);
                skm::BigIntOperField256::destory(tmp);
                bits->reset_size();

                if (nullptr != val) {
                    output->push(val);
                    break;
                }

            }
        }

        VmCache::returnBoolLst(bits);
        return output;
    }

    __device__ Vec<skm::BigInt*>* PoseidonGrainLFSR::get_field_elements_mod_p(u64 num_elems) {
        u64 num_bits = this->field_size_in_bits;
        Vec<bool>* bits = VmCache::getBoolLst(num_bits);
        Vec<u8>* bytes = VmCache::getU8Lst((num_bits + 7) / 8);
        Vec<skm::BigInt*>* output = VmCache::getBigIntLst(num_elems);
        for (int i = 0; i < num_elems; i++) {
            this->get_bits(bits, num_bits);
            bits->reverse();
            Vec<Vec<bool>*> *chunks = VmCacheVecOper::chunksBool(bits, 8);
            for (int j = chunks->getSize()-1; j >= 0; --j) {
                Vec<bool>* units = chunks->get(j);
                u8 sum = (u8)(units->get(0));
                u8 cur = 1;
                u8 skip = 1;
                for (int k = skip; k < units->getSize(); k++) {
                    cur *= 2;
                    sum += cur * ((u8)(units->get(k)));
                }
                bytes->push(sum);
            }
            VmCache::returnBoolLstLst(chunks);
            output->push(skm::BigIntOperField256::from_bytes_be_mod_order(bytes));
            bytes->reset_size();
            bits->reset_size();
        }

        VmCache::returnU8Lst(bytes);
        VmCache::returnBoolLst(bits);
        return output;
    }


    __device__  POSEIDON* POSEIDON::new_(const POSEIDON* src) {
        POSEIDON* rst = new POSEIDON();
        rst->RATE = src->RATE;
        rst->full_rounds = src->full_rounds;
        rst->partial_rounds = src->partial_rounds;
        rst->skip_matrices = src->skip_matrices;
        rst->domain = src->domain;
        rst->alpha = src->alpha;

        //初始化ark 和 mds
        rst->find_poseidon_ark_and_mds();
        return rst;
    }

    __device__ void POSEIDON::find_poseidon_ark_and_mds() {
        u64 full_rounds = this->full_rounds;
        u64 partial_rounds = this->partial_rounds;
        u64 skip_matrices = this->skip_matrices;
        u64 RATE = this->RATE;

        printf("******, %lld \n", full_rounds);
        printf("******, %lld \n", partial_rounds);
        printf("******, %lld \n", skip_matrices);
        printf("******, %lld \n", RATE);

        //253是一个field的大小
        PoseidonGrainLFSR *lfsr = PoseidonGrainLFSR::new_(false, skm::FIELD_SIZE, (RATE+1), full_rounds, partial_rounds);
        u64 ark_size = full_rounds + partial_rounds;
        Vec<Vec<skm::BigInt*>*>* ark = VmCache::getBigIntLstLst(ark_size);
        for (int i = 0; i < ark_size; i++) {
            ark->push(lfsr->get_field_elements_rejection_sampling(RATE+1));
        }
        for (int i = 0; i < skip_matrices; i++) {
            Vec<skm::BigInt*>* vecs = lfsr->get_field_elements_mod_p(2 * (RATE+1));
            VmCache::returnBigIntLst(vecs);
        }
        Vec<skm::BigInt*>* xs = lfsr->get_field_elements_mod_p(RATE + 1);
        Vec<skm::BigInt*>* ys = lfsr->get_field_elements_mod_p(RATE + 1);
        u64 mds_flattened_size = (RATE + 1) * (RATE + 1);
        Vec<skm::BigInt*>* mds_flattened = VmCache::getBigIntLst(mds_flattened_size);
        //填0
        for (int i = 0; i < mds_flattened_size; i++) {
            mds_flattened->push(skm::BigIntOperField256::clone(&skm::ZERO));
        }

        Vec<Vec<skm::BigInt*>*>* mds_row_i_s = VmCacheVecOper::chunksBigInt(mds_flattened, RATE + 1);;
        for (int i = 0; i < xs->getSize(), i < (RATE+1); i++) {
            skm::BigInt* x = xs->get(i);
            Vec<skm::BigInt*>* mds_row_i = mds_row_i_s->get(i);
            for (int j = 0; j < ys->getSize(), j < (RATE+1); j++) {
                skm::BigInt* y = ys->get(j);
                skm::BigIntOperField256::copy_to(skm::BigIntOperField256::add(x, y), mds_row_i->get(j));
            }
        }
        serial_batch_inversion_and_mul(mds_flattened, &skm::ONE);
        Vec<Vec<skm::BigInt*>*>* mds = VmCacheVecOper::chunksBigInt(mds_flattened, RATE + 1);
        this->ark = ark;
        this->mds = mds;
        delete(lfsr);
        VmCache::returnBigIntLstOnly(mds_flattened);
        VmCache::returnBigIntLst(xs);
        VmCache::returnBigIntLst(ys);
        VmCache::returnBigIntLstLstOnly(mds_row_i_s);
    }

    __device__ void POSEIDON::serial_batch_inversion_and_mul(Vec<skm::BigInt*>* v, const skm::BigInt* coeff) {
        Vec<skm::BigInt*>* prod = VmCache::getBigIntLst(v->getSize());
        skm::BigInt* tmp = skm::BigIntOperField256::clone(&skm::ONE);
        for (int i = 0; i < v->getSize(); i++) {
            if (skm::BigIntOperField256::is_eq(v->get(i), &skm::ONE)) { continue;}
            prod->push(skm::BigIntOperField256::clone(skm::BigIntOperField256::mul_change(tmp, v->get(i))));
        }

        skm::BigInt* _tmp = tmp;
        tmp = skm::BigIntOperField256::inverse(tmp);
        skm::BigIntOperField256::destory(_tmp);
//            skm::BigIntOperField256::print_hex(coeff);
        skm::BigIntOperField256::mul_change(tmp, coeff);
//        prod->insert(0, skm::BigIntOperField256::clone(&skm::ONE));
        for (int i = v->getSize()-1; i >= 0; i--) {
            if (skm::BigIntOperField256::is_eq(v->get(i), &skm::ZERO)) {
                continue;
            }
            skm::BigInt* f = v->get(i);
            skm::BigInt* s;
            if (i - 1 >=0) {
                s = prod->get(i - 1);//skip 一个
            } else {
                s = &skm::ONE;
            }
            skm::BigInt* new_tmp = skm::BigIntOperField256::mul(tmp, f);
            skm::BigInt* f_tmp = skm::BigIntOperField256::mul(tmp, s);
            skm::BigIntOperField256::copy_to(f_tmp, f);
            skm::BigIntOperField256::copy_to(new_tmp, tmp);
            skm::BigIntOperField256::destory(new_tmp);
            skm::BigIntOperField256::destory(f_tmp);
        }

        skm::BigIntOperField256::destory(tmp);
        VmCache::returnBigIntLst(prod);
    }
}