//
// Created by fil on 24-10-18.
//
#include "VmCacheVecOper.h"
#include "VmCache.cuh"

__device__  Vec<u8>* VmCacheVecOper::cloneU8(Vec<u8> *vec) {
    Vec<u8>* rst = VmCache::getU8Lst(vec->getSize());
    rst->setSize(vec->getSize());
    for (int i = 0; i < vec->getSize(); i++) {
        rst->set(i, vec->get(i));
    }
    return rst;
}

__device__  Vec<Bool> *VmCacheVecOper::cloneBoolStruct(Vec<Bool> *vec, int len) {
    Vec<Bool>* rst = VmCache::getBoolStructLst(len);
    rst->setSize(vec->getSize());
    for (int i = 0; i < vec->getSize(); i++) {
        rst->set(i, vec->get(i));
    }
    return rst;
}

__device__ Vec<Vec<Bool> *> *VmCacheVecOper::chunksBoolStruct(Vec<Bool> *vec, int chunk_size) {
    const int numChunks = (vec->getSize() + chunk_size - 1) / chunk_size; // 计算块的数量
    Vec<Vec<Bool> *> *newVecs = VmCache::getBoolStructLstLst(numChunks);
    Vec<Bool> *unit = nullptr;
    for (int i = 0; i < vec->getSize(); i++) {
        if (i % chunk_size == 0) {
            unit = VmCache::getBoolStructLst(chunk_size);
            newVecs->push(unit);
        }
        unit->push(vec->get(i));
    }
    return newVecs;
}

__device__ Vec<Vec<bool> *> *VmCacheVecOper::chunksBool(Vec<bool> *vec, int chunk_size) {
    const int numChunks = (vec->getSize() + chunk_size - 1) / chunk_size; // 计算块的数量
    Vec<Vec<bool> *> *newVecs = VmCache::getBoolLstLst(numChunks);
    Vec<bool> *unit = nullptr;
    for (int i = 0; i < vec->getSize(); i++) {
        if (i % chunk_size == 0) {
            unit = VmCache::getBoolLst(chunk_size);
            newVecs->push(unit);
        }
        unit->push(vec->get(i));
    }
    return newVecs;
}

__device__ Vec<Vec<skm::BigInt *> *> *VmCacheVecOper::chunksBigInt(Vec<skm::BigInt *> *vec, int chunk_size) {
    const int numChunks = (vec->getSize() + chunk_size - 1) / chunk_size; // 计算块的数量
    Vec<Vec<skm::BigInt *> *> *newVecs = VmCache::getBigIntLstLst(numChunks);
    Vec<skm::BigInt *> *unit = nullptr;
    for (int i = 0; i < vec->getSize(); i++) {
        if (i % chunk_size == 0) {
            unit = VmCache::getBigIntLst(chunk_size);
            newVecs->push(unit);
        }
        unit->push(vec->get(i));
    }
    return newVecs;
}

__device__ Vec<u8> *VmCacheVecOper::sub_array_change_u8(Vec<u8> *vec, int start_index) {
    Vec<u8>* rst = VmCache::getU8EmptyLst();
    return vec->sub_array_change(rst, start_index);
}

__device__ Vec<skm::BigInt *> *VmCacheVecOper::sub_array_change_big_int(Vec<skm::BigInt *> *vec, int start_index) {
    Vec<skm::BigInt *>* rst = VmCache::getBigIntEmptyLst();
    return vec->sub_array_change(rst, start_index);
}

__device__ Vec<Vec<u8> *> *VmCacheVecOper::split_at_u8(Vec<u8> *vec, int split_index) {
    Vec<Vec<u8>*>* rst = VmCache::getU8LstLst(2);
    Vec<u8>* r1 = VmCache::getU8Lst(split_index);
    Vec<u8>* r2 = VmCache::getU8Lst(vec->getSize() - split_index);
    for (int i = 0; i < vec->getSize(); i++) {
        if (i < split_index) {
            r1->push(vec->get(i));
        } else {
            r2->push(vec->get(i));
        }
    }
    rst->push(r1);
    rst->push(r2);
    return rst;
}
