//
// Created by fil on 24-10-23.
//

#ifndef SNARKVM_CUDA_BASE_ANY_CUH
#define SNARKVM_CUDA_BASE_ANY_CUH
#include "base_type.cuh"

class Any {
private:
    Type  type;
    union {
        u8 _u8;
        u16 _u16;
        u32 _u32;
        u64 _u64;
        skm::BigInt _u128;
        skm::BigInt _u256;
        i8 _i8;
        i16 _i16;
        i32 _i32;
        i64 _i64;
        skm::BigInt _i128;
        Bool _bool_struct;
        bool _bool;
    } data;
public:
    __device__ __forceinline__ Any();
    __device__ __forceinline__ Any(u8 val);
    __device__ __forceinline__ Any(u16 val);
    __device__ __forceinline__ Any(u32 val);
    __device__ __forceinline__ Any(u64 val);
    __device__ __forceinline__ Any(skm::BigInt val);
    __device__ __forceinline__ Any(i8 val);
    __device__ __forceinline__ Any(i16 val);
    __device__ __forceinline__ Any(i32 val);
    __device__ __forceinline__ Any(i64 val);
    __device__ __forceinline__ Any(Bool val);
    __device__ __forceinline__ Any(bool val);
    __device__ __forceinline__ Type getType();
    __device__ __forceinline__ Bool getBoolStruct();
    __device__ __forceinline__ skm::BigInt getBigInt();
    template<typename T>
    __device__ T get() {
        switch (type) {
            case Type::_u8: {return data._u8;}
            case Type::_u16: {return data._u16;}
            case Type::_u32: {return data._u32;}
            case Type::_u64: {return data._u64;}
            case Type::_i8: {return data._i8;}
            case Type::_i16: {return data._i16;}
            case Type::_i32: {return data._i32;}
            case Type::_i64: {return data._i64;}
            case Type::_bool: {return data._bool;}
        }
    }
};
__device__ __forceinline__ Any::Any() : type(Type::None) {}
__device__ __forceinline__ Any::Any(u8 val) {
    data._u8 = val;
    type = Type::_u8;
}
__device__ __forceinline__ Any::Any(u16 val) {
    data._u16 = val;
    type = Type::_u16;
}

__device__ __forceinline__ Any::Any(u32 val) {
    data._u32 = val;
    type = Type::_u32;
}

__device__ __forceinline__ Any::Any(u64 val) {
    data._u64 = val;
    type = Type::_u64;
}

__device__ __forceinline__ Any::Any(skm::BigInt val) {
    if (val.size == 2) { //128位
        if (val.is_signed) {
            data._i128 = val;
            type = Type::_i128;
        } else {
            data._u128 = val;
            type = Type::_u128;
        }
    } else if (val.size == 4) { //256位
        data._u256 = val;
        type = Type::_u256;
    }
}

__device__ __forceinline__ Any::Any(i8 val) {
    data._i8 = val;
    type = Type::_i8;
}

__device__ __forceinline__ Any::Any(i16 val) {
    data._i16 = val;
    type = Type::_i16;
}

__device__ __forceinline__ Any::Any(i32 val) {
    data._i32 = val;
    type = Type::_i32;
}

__device__ __forceinline__ Any::Any(i64 val) {
    data._i64 = val;
    type = Type::_i64;
}

__device__ skm::BigInt Any::getBigInt() {
    switch (type) {
        case Type::_u128: {return data._u128;}
        case Type::_u256: {return data._u256;}
        case Type::_i128: {return data._i128;}
    }
}

__device__ Type Any::getType() {
    return type;
}

__device__ Any::Any(Bool val) {
    data._bool_struct = val;
    type = Type::_bool_struct;
}

__device__ Any::Any(bool val) {
    data._bool = val;
    type = Type::_bool;
}

__device__ Bool Any::getBoolStruct() {
    switch (type) {
        case Type::_bool_struct: {
          return data._bool_struct;
       }
    }
}


#endif //SNARKVM_CUDA_BASE_ANY_CUH
