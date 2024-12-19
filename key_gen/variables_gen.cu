//
// Created by fil on 24-10-22.
//

#include "variables_gen.cuh"
#include "../base_cal/bigint.cuh"
#include "../base_cal/bigint_field256.cuh"

__device__ const u32 public_variable_size = 512;
__device__ const u32 pirvate_variable_size = 32768;

__device__ VariablesGen::VariablesGen() {
    public_variable = Vec<Any>::init(public_variable_size);
    private_variable = Vec<Any>::init(pirvate_variable_size);
}
__device__ VariablesGen::~VariablesGen() {
    delete(public_variable);
    delete(private_variable);
}

__device__ void VariablesGen::clear() {
    public_variable->reset_size();
    private_variable->reset_size();
}

__device__ void VariablesGen::public_gen(Vec<Any> *input) {
    for (int i = 0; i < input->getSize(); i++) {
        Any val = input->get(i);
        Type type = val.getType();
        switch (type) {
            case Type::_u128:
            case Type::_i128: {
                skm::BigInt v = val.getBigInt();
                Vec<Bool>* boolVec = skm::BigIntOper::to_bits_le(&v, Mode::Public);
                for (int j = 0; j < boolVec->getSize(); j++) {
                    public_variable->push(Any(Bool{boolVec->get(j).value, Mode::Public}));
                }
                break;
            }
            case Type::_u256: { //256位值，直接放入
                skm::BigInt v = val.getBigInt();
                v.mod = Mode::Public;
                public_variable->push(v);
                break;
            }
            case Type::_bool_struct: {
                public_variable->push(Any(Bool{val.getBoolStruct().value, Mode::Public}));
                break;
            }
            //以下是基本类型
            case Type::_bool: {
                public_variable->push(Any(Bool{val.get<bool>(), Mode::Public}));
                break;
            }
            case Type::_i8: {
                put_public_variable(public_variable, val.get<i8>());
                break;
            }
            case Type::_i16: {
                put_public_variable(public_variable, val.get<i16>());
                break;
            }
            case Type::_i32: {
                put_public_variable(public_variable, val.get<i32>());
                break;
            }
            case Type::_i64: {
                put_public_variable(public_variable, val.get<i64>());
                break;
            }
            case Type::_u8: {
                put_public_variable(public_variable, val.get<u8>());
                break;
            }
            case Type::_u16: {
                put_public_variable(public_variable, val.get<u16>());
                break;
            }
            case Type::_u32: {
                put_public_variable(public_variable, val.get<u32>());
                break;
            }
            case Type::_u64: {
                put_public_variable(public_variable, val.get<u64>());
                break;
            }
        }
    }
}

__device__  Vec<Any> *VariablesGen::getPublic() {
    return this->public_variable;
}





