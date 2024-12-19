//
// Created by fil on 24-10-17.
//

#ifndef SNARKVM_CUDA_BASE_TYPE_CUH
#define SNARKVM_CUDA_BASE_TYPE_CUH
typedef __int8_t I8;
typedef __int16_t I16;
typedef __int32_t I32;
typedef __int64_t I64;
typedef __int128_t I128;

typedef __uint8_t U8;
typedef __uint16_t U16;
typedef __uint32_t U32;
typedef __uint64_t U64;
typedef __uint128_t U128;

typedef __int8_t i8;
typedef __int16_t i16;
typedef __int32_t i32;
typedef __int64_t i64;
typedef __int128_t i128;

typedef __uint8_t u8;
typedef __uint16_t u16;
typedef __uint32_t u32;
typedef __uint64_t u64;
typedef __uint128_t u128;
//====================================普通常量=================================================
__device__ __constant__ extern u8 MAX_U8;
__device__ __constant__ extern u16 MAX_U16;
__device__ __constant__ extern u32 MAX_U32;
__device__ __constant__ extern u64 MAX_U64;
__device__ __constant__ extern i8 MAX_I8;
__device__ __constant__ extern i16 MAX_I16;
__device__ __constant__ extern i32 MAX_I32;
__device__ __constant__ extern i64 MAX_I64;
__device__ __constant__ extern i8 MIN_I8;
__device__ __constant__ extern i16 MIN_I16;
__device__ __constant__ extern i32 MIN_I32;
__device__ __constant__ extern i64 MIN_I64;
__device__ __constant__ extern U128 MAX_U128;
__device__ __constant__ extern I128 MAX_I128;
__device__ __constant__ extern I128 MIN_I128;

// 通用模板，默认为false
template<typename T>
__device__ bool isPointer(T) {
    return false;
}

// 特化用于指针类型
template<typename T>
__device__ bool isPointer(T*) {
    return true;
}

enum Mode {
    Constant,
    Public,
    Private,
};

struct Bool {
    bool value;
    Mode mod;
};

template<typename T>
struct VecStruct {
    int size;
    int capacity;
    T* val;
};

//存储任意类型数据
//todo 未能成功实现
class Data {
private:
    void *ptr{nullptr};
public:
    __device__ Data(){
    }
    template <typename T>
    __device__ Data(T data){
        set(data);
    }
    template <typename T>
    __device__ void set(T data) {
        if (isPointer(data)) {
            ptr = &data;
        } else {
            T* t = new T;
            *t = data;
            ptr = t;
        }
    }

    template <typename T>
    __device__ T get() {
        T *res_ptr{(T *) ptr};
        return *res_ptr;
    }
};


// 哈希表项 为了处理简单，key使用int
template<typename K, typename T>
struct HashItem {
    K key;   // 存储的键
    T value; // 存储的值
    bool occupied; // 记录该位置是否被占用
};

namespace skm {
    constexpr int MAX_DIGITS = 4;
    struct BigInt {
        int size;
        bool is_signed;
        //从左到右为从低位到高位
        u64 number[MAX_DIGITS];
        Mode mod;
        __device__ void reset() {
            size = 0;
            is_signed = false;
            mod = Mode::Private;
            for (int i = 0; i < MAX_DIGITS; i++) {
               number[i] = 0;
            }
        }

        operator int() const {  // 定义一个从 MyStruct 到 int 的转换操作符
            return int(number[0] && MAX_U32);  // 任何需要的逻辑
        }
    };
}

enum class Type { _u8, _u16, _u32, _u64, _u128, _u256, _i8, _i16, _i32, _i64, _i128,_bool,_bool_struct, None};

#endif //SNARKVM_CUDA_BASE_TYPE_CUH
