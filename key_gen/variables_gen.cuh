//
// Created by fil on 24-10-22.
//

#ifndef SNARKVM_CUDA_VARIABLES_GEN_CUH
#define SNARKVM_CUDA_VARIABLES_GEN_CUH
#include "../base_cal/base_type.cuh"
#include "../base_cal/base_vec.cuh"
#include "../base_cal/base_any.cuh"
#include "../base_cal/basic_oper.cuh"
class VariablesGen {
private:
    Vec<Any>* public_variable;
    Vec<Any>* private_variable;
    template<typename T>
    __device__ void put_public_variable(Vec<Any>* public_var, T normalVal);
public:
    __device__ VariablesGen();
    __device__ ~VariablesGen();
    __device__ void clear();
    __device__ void public_gen(Vec<Any>* input);
    __device__  Vec<Any>* getPublic();
};

template<typename T>
__device__ void VariablesGen::put_public_variable(Vec<Any> *public_var, T normalVal) {
    Vec<Bool>* boolVec = skm::BasicOper::to_bits_le(normalVal);
    for (int j = 0; j < boolVec->getSize(); j++) {
        public_var->push(Any(Bool{boolVec->get(j).value, Mode::Public}));
    }
}


#endif //SNARKVM_CUDA_VARIABLES_GEN_CUH
