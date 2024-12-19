//
// Created by fil on 24-10-16.
//

#ifndef SNARKVM_CUDA_TEST_VM_CACHE_CUH
#define SNARKVM_CUDA_TEST_VM_CACHE_CUH
#include "../base_cal/cache/VmCache.cuh"
namespace test_vm_cache {
    //ok
    __device__ void test_getInstance() {
//        vm_cache* cache = vm_cache::getInstance();
//        vm_cache* cache2 = vm_cache::getInstance();
//        printf("%d\n", cache == cache2);
    }

    __device__ void test_createBigIntLst() {
//        vm_cache* cache = vm_cache::getInstance();
//        cache->init();
//        printf("=========");
    }
}
#endif //SNARKVM_CUDA_TEST_VM_CACHE_CUH
