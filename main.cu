#include <stdio.h>
#include <cuda_runtime.h>
//#include "base_cal/bigint_field256.cuh"
//#include "data_generate/input_generator.cuh"
//#include "base_cal/cache/vm_cache.cuh"
#include "test/test_poseidon_hash.cuh"

#include "test/test_bigint_field256.cuh"
#include "test/test_basic_oper.cuh"
//#include "test/test_bigint.cuh"
#include "test/test_vm_cache.cuh"
#include "test/test_variables_gen.cuh"


class TestClass {
public:
    template<typename T>
    __device__  T device_add(T a, T b);
};

template<typename T>
__device__  T TestClass::device_add(T a11, T b11) {
//    printf("1111: %d\n", std::is_same<HashMap<HashMap<Vec<Vec<Bool> *> *> *> *, HashMap<HashMap<Vec<Vec<Bool> *> *> *> *>::value);
    //=============测试==================
//    test_bigint::test_shl_w(); //左移
//    test_bigint::test_shr_w();  //右移
//      test_bigint::test_div_w();  //除法
//    test_poseidon::test_psd2();   //psd hash
//      test_basic_oper::testInt128();

//    test_basic_oper::testData();
      test_variables_gen::test_public_variable();

//    test_bingint_field256::test_from_bit_le();
//    test_bingint_field256::test_field256_pow(); //乘方
//    test_basic_oper::test_new_domain_separator();
//    test_basic_oper::test_add_head();
//    test_vm_cache::test_vm_cache();
//    test_vm_cache::test_createBigIntLst();

    return skm::BasicOper::add(a11, b11);
}

__global__ void kernel(TestClass* obj) {

    VmCache::init();
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int result = obj->device_add(thread_id, thread_id);

    // 可以在这里做一些有趣的操作，比如将结果写回到全局内存中
//    large_int_skm::BigIntOperField256::init();
}

int main() {
    void* device_memory = nullptr;
    TestClass* host_obj;
    TestClass* device_obj;
    cudaMalloc((void**)&device_obj, 1024*1024*1024);
    cudaMemcpy(device_obj, &host_obj, sizeof(TestClass), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(device_obj);
    printf("~~~~\n");
    cudaError_t error = cudaGetLastError();
    if (cudaSuccess != error) {
        printf("错误: %s,,,%s", cudaGetErrorName(error), cudaGetErrorString(error));
    }
    cudaFree(device_obj);
    return 0;
}
