//
// Created by fil on 24-10-15.
//

#ifndef SNARKVM_CUDA_TEST_BASIC_OPER_CUH
#define SNARKVM_CUDA_TEST_BASIC_OPER_CUH
#include "../base_cal/bigint_field256.cuh"
#include "../base_cal/base_map.cuh"
#include "../base_cal/base_any.cuh"

namespace test_basic_oper {
    //ok
    __device__ void test_new_domain_separator() {
        char* str = "AleoPoseidon2";
        skm::BigInt* a = skm::BigIntOperField256::new_domain_separator(str);
        skm::BigIntOperField256::print_hex(a);
    }

    class Test {
    private:
        HashMap<int, HashMap<int, Vec<Vec<bool> *> *> *> *vecBoolLst;
        HashMap<int, Vec<Vec<bool> *> *> * type;
        __device__  int getGrade(int len) {
            int grade = 0;
            if (len <= cache_type_len[0]) {
                //不用处理
            } else if (len > cache_type_len[0] && len <= cache_type_len[1]) {
                grade = 1;
            } else if (len > cache_type_len[1] && len <= cache_type_len[2]) {
                grade = 2;
            } else if (len > cache_type_len[2] && len <= cache_type_len[3]) {
                grade = 3;
            } else if (len > cache_type_len[3] && len <= cache_type_len[4]) {
                grade = 4;
            } else {
                grade = 5;
            }
            return grade;
        }

    public:
        __device__ Test() {
            if (nullptr == this->vecBoolLst) {
                this->vecBoolLst = new HashMap<int, HashMap<int, Vec<Vec<bool> *> *> *>(1024);
            }
            this->type = new HashMap<int, Vec<Vec<bool> *> *>(cache_type_size);
            for (int i = 0; i < cache_type_size; i++) {
                Vec<Vec<bool> *> *vec = Vec<Vec<bool>*>::init(cache_type_amount[i]);
                for (int j = 0; j < cache_type_amount[i]; j++) {
                    Vec<bool> *newVec = Vec<bool>::init(cache_type_len[i]);
                    vec->push(newVec);
                }
                type->insert(i, vec);
            }
            this->vecBoolLst->insert(0, type);
        }


        __device__ Vec<bool> *get(int len) {
            Vec<Vec<bool> *> *vec = type->find(0);
            if (vec->getSize() != 0) { //有值，从缓存里面取
                Vec<bool> *r = vec->pop();
                return r;
            }
            Vec<bool> *r = Vec<bool>::init(cache_type_len[0]); //没有值，则新建一个
            return r;
        }

        __device__ void returnObj(Vec<bool> *vec) {
            if (nullptr == vec) {
                return;
            }
            int len = vec->getSize();
            int grade = getGrade(vec->getCapacity());
            vec->reset_size();
            Vec<Vec<bool> *> *array = type->find(0);
            array->push(vec);
            for (int i = 0; i < vec->getCapacity(); i++) { //置0
                vec->set(i, false);
            }
        };
    };

    //ok
    __device__ void test_map() {
        Test* test = new Test();
        u64 index = 0;
        while (true) {
            index++;
            Vec<bool>* vec = test->get(4);
            test->returnObj(vec);
            if (index % 10000000 == 0) {
               printf("%lld\n", index);
            }
        }
    }

    __device__ void testInt128() {

    }



    __device__ void testData() {
        bool a = false;
        u8 b = (u8)a;
        printf("%d\n", b);
    }
}


#endif //SNARKVM_CUDA_TEST_BASIC_OPER_CUH
