//
// Created by fil on 24-10-16.
//

#include <cuda_runtime.h>
#include "hash_poseidon.cuh"
#include "../base_cal/cache/VmCache.cuh"
#include "../base_cal/cache/VmCacheVecOper.h"
namespace hash_poseidon {
    //===========================重要常数 start=================================
    __device__ __constant__ PoseidonDefaultParametersEntry poseidonDefaultParametersFq256[] = {
            PoseidonDefaultParametersEntry{2, 17, 8, 31, 0},
            PoseidonDefaultParametersEntry{3, 17, 8, 31, 0},
            PoseidonDefaultParametersEntry{4, 17, 8, 31, 0},
            PoseidonDefaultParametersEntry{5, 17, 8, 31, 0},
            PoseidonDefaultParametersEntry{6, 17, 8, 31, 0},
            PoseidonDefaultParametersEntry{7, 17, 8, 31, 0},
            PoseidonDefaultParametersEntry{8, 17, 8, 31, 0},
    };
    __device__ __constant__ u32 poseidonDefaultParametersFq256Size = 7;
    __device__ __constant__ U8 CAPACITY = 1;
    __device__ __constant__ u32 poseidon2_rate = 2;
    __device__ __constant__ u32 poseidon4_rate = 4;
    __device__ __constant__ u32 poseidon8_rate = 8;
    __device__ __constant__ char* DomainAleoPoseidon2  = "AleoPoseidon2";
    __device__ __constant__ char* DomainAleoPoseidon4  = "AleoPoseidon4";
    __device__ __constant__ char* DomainAleoPoseidon8  = "AleoPoseidon8";

    //===========================重要常数 end=================================

    //全局变量
    __device__  ark_and_mds_pos::POSEIDON *POSEIDON_2 = nullptr;
    __device__ int pose_is_init = 0;

    __device__ bool Range::contains(int* i) {
        if (*i >= start && *i < end) {
            return true;
        }
        return false;
    }

    __device__ void apply_ark(ark_and_mds_pos::POSEIDON *poseidon, Vec<skm::BigInt *> *state, U64 round) {
        for (int i = 0; i < state->getSize(); i++) {
            skm::BigIntOperField256::add_change(state->get(i), poseidon->ark->get(round)->get(i));
        }
    }

    __device__ void apply_s_box(ark_and_mds_pos::POSEIDON *poseidon, Vec<skm::BigInt *> *state, bool is_full_round) {
        if (is_full_round) {
            for (int i = 0; i < state->getSize(); i++) {
                skm::BigInt *tmp = skm::BigIntOperField256::pow(state->get(i), poseidon->alpha);
                skm::BigIntOperField256::copy_to(tmp, state->get(i));
                skm::BigIntOperField256::destory(tmp);
            }

        } else {
            // Partial rounds apply the S Box (x^alpha) to just the first element of state
            skm::BigInt *tmp = skm::BigIntOperField256::pow(state->get(0), poseidon->alpha);
            skm::BigIntOperField256::copy_to(tmp, state->get(0));
            skm::BigIntOperField256::destory(tmp);
        }
    }

    __device__ void apply_mds(ark_and_mds_pos::POSEIDON *poseidon, Vec<skm::BigInt *> *state) {
        Vec<skm::BigInt *> *new_state = VmCache::getBigIntLst(state->getSize());
        for (int i = 0; i < state->getSize(); i++) {
            skm::BigInt *accumulator = skm::BigIntOperField256::clone(&skm::ZERO);
            for (int j = 0; j < state->getSize(); j++) {
                skm::BigInt *mul_val = skm::BigIntOperField256::mul(state->get(j), poseidon->mds->get(i)->get(j));
                skm::BigIntOperField256::add_change(accumulator, mul_val);
                skm::BigIntOperField256::destory(mul_val);
            }
            new_state->push(accumulator);
        }
        skm::BigIntOperField256::clone_from_slice(state, new_state);
        VmCache::returnBigIntLst(new_state);
    }

    __device__ void permute(ark_and_mds_pos::POSEIDON *poseidon, Vec<skm::BigInt *> *state) {
        u64 full_rounds_over_2 = poseidon->full_rounds / 2;
        Range partial_round_range = {full_rounds_over_2, full_rounds_over_2 + poseidon->partial_rounds};
        for (int i = 0; i < (poseidon->partial_rounds + poseidon->full_rounds); i++) {
            bool is_full_round = !partial_round_range.contains(&i);
            apply_ark(poseidon, state, i);
            apply_s_box(poseidon, state, is_full_round);
            apply_mds(poseidon, state);
        }
    }

    __device__ void absorb(ark_and_mds_pos::POSEIDON *self, Vec<skm::BigInt *> *state, DuplexSpongeMode *mode,
                           Vec<skm::BigInt *> *input) {
        if (input->getSize() == 0) {
            return;
        }
        bool should_permute = false;
        U64 absorb_index = 0;
        if (DuplexSpongeModeEnum::Absorbing == mode->duplexSpongeMode) {
            if (mode->next_index == self->RATE) {
                absorb_index = 0;
                should_permute = true;
            } else {
                absorb_index = mode->next_index;
                should_permute = false;
            }
        } else {
            absorb_index = 0;
            should_permute = true;
        }

        if (should_permute) {
            permute(self, state);
        }
        Vec<skm::BigInt *> *remaining = input;
        while (true) {
            U64 start = (U64) CAPACITY + absorb_index;
            if (absorb_index + remaining->getSize() <= self->RATE) {
                for (int i = 0; i < remaining->getSize(); i++) {
                    skm::BigIntOperField256::add_change(state->get(start + i), remaining->get(i));
                }
                mode->duplexSpongeMode = DuplexSpongeModeEnum::Absorbing;
                mode->next_index = absorb_index + remaining->getSize();
                if (remaining != input) { VmCache::returnBigIntEmptyLstOnly(remaining); }
                return;
            }
            u64 num_absorbed = self->RATE - absorb_index;
            for (int i = 0; i < num_absorbed && i < remaining->getSize(); i++) {
                skm::BigIntOperField256::add_change(state->get(start + i), remaining->get(i));
            }
            permute(self, state);
            Vec<skm::BigInt *> *tmp = remaining;
            remaining = VmCacheVecOper::sub_array_change_big_int(remaining, num_absorbed);
            if (tmp != input) {
                VmCache::returnBigIntEmptyLstOnly(tmp);
            }
            absorb_index = 0;
        }
    }

    __device__ void squeeze_internal(ark_and_mds_pos::POSEIDON *self, Vec<skm::BigInt *> *state, DuplexSpongeMode *mode,
                                     Vec<skm::BigInt *> *output) {
        u64 squeeze_index = 0;
        bool should_permute = false;
        if (mode->duplexSpongeMode == DuplexSpongeModeEnum::Absorbing) {
            squeeze_index = 0;
            should_permute = true;
        } else {
            if (self->RATE == mode->next_index) {
                squeeze_index = 0;
                should_permute = true;
            } else {
                squeeze_index = mode->next_index;
                should_permute = false;
            }
        }
        if (should_permute) {
            permute(self, state);
        }
        Vec<skm::BigInt *> *remaining = output;

        while (true) {
            u64 start = CAPACITY + squeeze_index;
            if ((squeeze_index + remaining->getSize()) <= self->RATE) {
                skm::BigIntOperField256::clone_from_slice(remaining, state, start, remaining->getSize());

                // Update the sponge mode.
                mode->duplexSpongeMode = DuplexSpongeModeEnum::Squeezing;
                mode->next_index = squeeze_index + remaining->getSize();
                if (remaining != output) { VmCache::returnBigIntEmptyLstOnly(remaining); }
                return;
            }

            // Otherwise, proceed to squeeze `(rate - squeeze_index)` elements.
            u64 num_squeezed = self->RATE - squeeze_index;
            skm::BigIntOperField256::clone_from_slice(remaining, state, start, num_squeezed);

            // Permute.
            permute(self, state);

            // Repeat with the updated output slice and squeeze index.
            Vec<skm::BigInt *> *tmp = remaining;
            remaining = VmCacheVecOper::sub_array_change_big_int(remaining, num_squeezed);
            if (tmp != output) {
                VmCache::returnBigIntEmptyLstOnly(tmp);
            }
            squeeze_index = 0;
        }
    }

    __device__ Vec<skm::BigInt *> *
    squeeze(ark_and_mds_pos::POSEIDON *self, Vec<skm::BigInt *> *state, DuplexSpongeMode *mode, const u64 num_outputs) {
        Vec<skm::BigInt *> *output = VmCache::getBigIntLst(num_outputs);
        for (int i = 0; i < num_outputs; i++) {
            output->push(skm::BigIntOperField256::clone(&skm::ZERO));
        }

        if (0 != num_outputs) {
            squeeze_internal(self, state, mode, output);
        }

        return output;
    }

    __device__ Vec<skm::BigInt *> *
    hash_many(ark_and_mds_pos::POSEIDON *self, Vec<skm::BigInt *> *inputs, u64 num_outputs) {
        int len = inputs->getSize();
        Vec<skm::BigInt *> *preimage = VmCache::getBigIntLst(len + self->RATE + 1);
        preimage->push(skm::BigIntOperField256::clone(self->domain));
        skm::BigInt *tmp = skm::BigIntOperField256::CreatInt256(len, 0, 0, 0, false, Mode::Constant);
        preimage->push(skm::BigIntOperField256::from_bigint(tmp));
        skm::BigIntOperField256::destory(tmp);
        if (self->RATE > preimage->getSize()) { //重置数组长度
            for (int i = 0; i < (self->RATE - preimage->getSize()); i++) {
                preimage->push(skm::BigIntOperField256::clone(&skm::ZERO));
            }
        }
        if (self->RATE < preimage->getSize()) { //重置数组长度
            for (int i = 0; i < (preimage->getSize() - self->RATE); i++) {
                skm::BigIntOperField256::destory(preimage->pop());
            }
        }
        for (int i = 0; i < inputs->getSize(); i++) {
            preimage->push(skm::BigIntOperField256::clone(inputs->get(i)));
        }
        Vec<skm::BigInt *> *state = VmCache::getBigIntLst(self->RATE + CAPACITY);
        for (int i = 0; i < self->RATE + CAPACITY; i++) {
            state->push(skm::BigIntOperField256::clone(&skm::ZERO));
        }
        DuplexSpongeMode mode{DuplexSpongeModeEnum::Absorbing, 0};
        absorb(self, state, &mode, preimage);
        Vec<skm::BigInt *> *rst = squeeze(self, state, &mode, num_outputs);
        VmCache::returnBigIntLst(preimage);
        VmCache::returnBigIntLst(state);

        return rst;
    }

    __device__ skm::BigInt *hash(ark_and_mds_pos::POSEIDON *self, Vec<skm::BigInt *> *inputs) {
        Vec<skm::BigInt *> *rstLst = hash_many(self, inputs, 1);
        skm::BigInt *rst = rstLst->swap_remove(0);
        VmCache::returnBigIntLst(rstLst);
        return rst;
    }

    __device__ void init_hash_psd2() {
        u64 RATE = poseidon2_rate; // psd2   不同的psd改为不同的值
        char *domain = DomainAleoPoseidon2;  //不同domain改为不同的值
        ark_and_mds_pos::POSEIDON *pos = new ark_and_mds_pos::POSEIDON();
        pos->domain = skm::BigIntOperField256::new_domain_separator(domain);
        for (int i = 0; i < poseidonDefaultParametersFq256Size; i++) {
            if (poseidonDefaultParametersFq256[i].rate == RATE) {
                pos->RATE = RATE;
                skm::BigInt *tmp = skm::BigIntOperField256::CreatInt256(poseidonDefaultParametersFq256[i].alpha, 0, 0,
                                                                        0, false, Mode::Constant);
                skm::BigInt *alpha = skm::BigIntOperField256::from_bigint(tmp);
                skm::BigIntOperField256::destory(tmp);
                pos->alpha = alpha;
                pos->full_rounds = poseidonDefaultParametersFq256[i].full_rounds;
                pos->partial_rounds = poseidonDefaultParametersFq256[i].partial_rounds;
                pos->skip_matrices = poseidonDefaultParametersFq256[i].skip_matrices;
                break;
            }
        }

        if (nullptr == POSEIDON_2) {
            POSEIDON_2 = ark_and_mds_pos::POSEIDON::new_(pos);
        }
        delete (pos);
    }

    __device__ skm::BigInt *hash_psd2(Vec<skm::BigInt *> *inputs) {
        if (nullptr == POSEIDON_2) {
            // 获取锁
            skm::acquire_lock(&pose_is_init);
            if (nullptr == POSEIDON_2) {
                init_hash_psd2();
            }
            // 释放锁
            skm::release_lock(&pose_is_init);
        }
        return hash(POSEIDON_2, inputs);
    }
}