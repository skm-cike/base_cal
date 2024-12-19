//
// Created by fil on 24-10-17.
//
#include "basic_oper.cuh"
#include "cache/VmCache.cuh"
namespace  skm {
    __device__  Vec<u8>* BasicOper::as_bytes(const char* a) {
        int len = stringLength(a);
        Vec<u8>* rst = VmCache::getU8Lst(len);
        for (int i = 0; i < len; i++) {
            rst->push((u8)a[i]);
        }
        return rst;
    }

    __device__ int BasicOper::stringLength(const char* str) {
        int length = 0;
        while (str[length] != '\0') {
            length++;
        }
        return length;
    }

    __device__ void acquire_lock(int *lock) {
        while (atomicCAS(lock, 0, 1) != 0);  // 如果不为0，说明锁已经被占用，线程等待
    }

    __device__ void release_lock(int *lock) {
        atomicExch(lock, 0);  // 释放锁，将锁设置为0
    }
}