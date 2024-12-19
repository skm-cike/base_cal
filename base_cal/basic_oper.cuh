#pragma once
#ifndef CUDA_SNARKVM_BASICOPER_H
#define CUDA_SNARKVM_BASICOPER_H

#include "cache/VmCache.cuh"
#include "base_vec.cuh"

namespace  skm {
    template<typename T>
    class BaseTypeLimit {
    public:
        __device__ static T getMax();
    };
    class BasicOper {
        public:
            template<typename T>
            __device__ static Vec<u8>* to_le_bytes(Vec<u8>* vec, T value);
            template<typename T>
            __device__ static int max(T len1, T len2);
            template<typename T>
            __device__ static T checked_shl(T num, int shift_amount);
            template<typename T>
            __device__ static int min(T len1, T len2);
            template<typename T>
            __device__ static void swap(T* a, T* b);
            template<typename T>
            __device__ static T mul_w(T a, T b);
            template<typename T>
            __device__ static T mul(T a, T b);
            template<typename T>
            __device__ static T add_w(T a, T b);
            template<typename T>
            __device__ static T add(T a, T b);
            template<typename T>
            __device__ static bool is_eq(T a, T b);
            template<typename T>
            __device__ static bool is_neq(T a, T b);
            template<typename T>
            __device__ static T ternary(bool condition, T a, T b);
            template<typename T>
            __device__ static bool lt(T a, T b);
            template<typename T>
            __device__ static bool lte(T a, T b);
            template<typename T>
            __device__ static bool gt(T a, T b);
            template<typename T>
            __device__ static bool gte(T a, T b);
            //逻辑或
            template<typename T>
            __device__ static T xor_(T a, T b);
            template<typename T>
            __device__ static T abs_w(T a);
            template<typename T>
            __device__ static T abs(T a);
            template<typename T>
            __device__ static T neg(T a);
            template<typename T>
            __device__ static T pow_w(T base, T exponent);
            //逻辑与
            template<typename T>
            __device__ static T and_(T a, T b);
            template<typename T>
            __device__ static T square(T a);
            //左移多少位，只能是U32类型
            template<typename T>
            __device__ static T shl_w(T value, U32 shift);
            //右移，只能是U32类型
            template<typename T>
            __device__ static T shr_w(T value, U32 shift);
            // 模板函数，允许精度丢失，适用于 CUDA 设备端
            template<typename From, typename To>
            __device__
            static typename std::enable_if<std::is_integral<From>::value && std::is_integral<To>::value, To>::type
            cast_lossy(From from);
            template<typename T>
            __device__ static void print_binary(T num);
            template<typename T>
            __device__ static Vec<Bool>* to_bits_le(Vec<Bool>* vec, T num);
            template<typename T>
            __device__ static Vec<Bool>* to_bits_le(T num);
            __device__ static Vec<u8>* as_bytes(const char* a);
            __device__ static int stringLength(const char* str);
            template<typename T>
            __device__ static void print_hex(T num);
            __device__ static bool nand(bool a, bool b);
    };

//===========================================实现================================================
    template<typename T>
    __device__ T BaseTypeLimit<T>::getMax() {
        bool is_unsigned = std::is_unsigned<T>::value;
        int size = sizeof(T);
        if (is_unsigned) {
            switch (size) {
                case 1: {return MAX_U8;}
                case 2: {return MAX_U16;}
                case 4: {return MAX_U32;}
                case 8: {return MAX_U64;}
                default: {return 0;}
            }
        } else {
            switch (size) {
                case 1: {return MAX_I8;}
                case 2: {return MAX_I16;}
                case 4: {return MAX_I32;}
                case 8: {return MAX_I64;}
                default: {return 0;}
            }
        }
    }

    __device__ inline bool BasicOper::nand(bool a, bool b) {
        if (a && b) {
            return false;
        }
        return true;
    }

    template<typename T>
    __device__  Vec<u8>* BasicOper::to_le_bytes(Vec<u8>* vec, T value) {
        int len = sizeof(T);
        for (int i = 0; i < len; i++) {
            vec->push(static_cast<u8>(value & 0xFF));
            value >>= 8;
        }
        return vec;
    }

    template<typename T>
    __device__  int BasicOper::max(T len1, T len2) {
        return len1>len2?len1:len2;
    }

    template<typename T>
    __device__  T BasicOper::checked_shl(T num, int shift_amount) {
        // 检查移位是否会导致溢出
        if (shift_amount >= sizeof(num) * CHAR_BIT) {
            return (T)0; // 溢出，返回 0
        }
        T shifted = num << shift_amount;
        // 检查移位结果是否小于原始数
        if (shifted < num) {
            return (T)0; // 溢出，返回 0
        }
//                *result = shifted;
        return (T)1; // 成功，返回 1
    }
    template<typename T>
    __device__  int BasicOper::min(T len1, T len2) {
        return len1>len2?len2:len1;
    }
    template<typename T>
    __device__  void BasicOper::swap(T* a, T* b) {
        T tmp = *a;
        *a = *b;
        *b = tmp;
    }
    template<typename T>
    __device__  T BasicOper::mul_w(T a, T b) {
        return (T) (a * b);
    }

    template<typename T>
    __device__  T BasicOper::mul(T a, T b) {
        return a * b;
    }

    template<typename T>
    __device__  T BasicOper::add_w(T a, T b) {
        return (T) (a + b);
    }
    template<typename T>
    __device__  T BasicOper::add(T a, T b) {
        return a + b;
    }
    template<typename T>
    __device__  bool BasicOper::is_eq(T a, T b) {
        return a == b;
    }
    template<typename T>
    __device__  bool BasicOper::is_neq(T a, T b) {
        return a != b;
    }

    template<typename T>
    __device__  T BasicOper::ternary(bool condition, T a, T b) {
        condition?a:b;
    }

    template<typename T>
    __device__  bool BasicOper::lt(T a, T b) {
        return a < b;
    }

    template<typename T>
    __device__  bool BasicOper::lte(T a, T b) {
        return a <= b;
    }

    template<typename T>
    __device__  bool BasicOper::gt(T a, T b) {
        return a > b;
    }

    template<typename T>
    __device__  bool BasicOper::gte(T a, T b) {
        return a >= b;
    }

    //逻辑或
    template<typename T>
    __device__  T BasicOper::xor_(T a, T b) {
        return a ^ b;
    }
    template<typename T>
    __device__  T BasicOper::abs_w(T a) {
        // Manually define the minimum value for common integer types
        if constexpr (std::is_same<T, I8>::value) {
            constexpr T min_value = -(1 << 7);
            if (a == min_value) return min_value;
        } else if constexpr (std::is_same<T, I16>::value) {
            constexpr T min_value = -(1 << 15);
            if (a == min_value) return min_value;
        } else if constexpr (std::is_same<T, I32>::value) {
            constexpr T min_value = -(1 << 31); // Minimum value for int: -2^31
            if (a == min_value) return min_value;
        } else if constexpr (std::is_same<T, I64>::value) {
            constexpr T min_value = -(1LL << 63); // Minimum value for long/long long: -2^63
            if (a == min_value) return min_value;
        }
        // General case: return the absolute value
        return (a < 0) ? -a : a;
        //todo 待处理128整型的绝对值
    }
    template<typename T>
    __device__  T BasicOper::abs(T a) {
        return abs_w(a);
    }

    template<typename T>
    __device__  T BasicOper::neg(T a) {
        return -a;
    }

    template<typename T>
    __device__  T BasicOper::pow_w(T base, T exponent) {
        T max = skm::BaseTypeLimit<T>::getMax();
        T result = 1;
        T temp = base;
        while (exponent > 0) {
            if (exponent & 1) {
                result *= temp;
                result %= max; // Handle overflow
            }
            exponent >>= 1; // Move to next bit
            temp *= temp;
            temp %= max; // Handle overflow
        }
        return result;
    }

    //逻辑与
    template<typename T>
    __device__  T BasicOper::and_(T a, T b) {
        return a & b;
    }

    template<typename T>
    __device__  T BasicOper::square(T a) {
        return a * a;
    }

    //左移多少位，只能是U32类型
    template<typename T>
    __device__  T BasicOper::shl_w(T value, U32 shift) {
        T symbol = 1;
        if (value < 0) {
            symbol = -1;
            value = -value;
        }
        U32 bit_width = sizeof(T) * 8;
//            T max = skm_aleo_cuda::BaseTypeLimit<T>::getMax();
        T max = (T(1) << bit_width) - 1;
        shift %= bit_width;
        T shifted_value = (value << shift) & max;
//            print_binary(shifted_value);
        return shifted_value * symbol;
    }

    //右移，只能是U32类型
    template<typename T>
    __device__  T BasicOper::shr_w(T value, U32 shift) {
        //todo 右移
    }

    // 模板函数，允许精度丢失，适用于 CUDA 设备端
    template<typename From, typename To>
    __device__
    typename std::enable_if<std::is_integral<From>::value && std::is_integral<To>::value, To>::type
    BasicOper::cast_lossy(From from) {
        return static_cast<To>(from);  // 直接进行转换，允许精度丢失
    }

    template<typename T>
    __device__  void BasicOper::print_binary(T num) {
        for (int i = sizeof(T) * 8 - 1; i >= 0; --i) {
            int bit = (num >> i) & 1;
            printf("%d", bit);
        }
        printf("\n");
    }

    template<typename T>
    __device__  Vec<Bool>* BasicOper::to_bits_le(Vec<Bool>* vec, T num) {
        int len = sizeof(T) * 8;
        T val = num;
        for (int i = 0; i < len; i++) {
            vec->push(Bool{bool(val & 0x1), Mode::Private});
            val = val >> 1;
        }
        return vec;
    }

    template<typename T>
    __device__  Vec<Bool>* BasicOper::to_bits_le(T num) {
        int len = sizeof(T) * 8;
        Vec<Bool>* vec = Vec<Bool>::init(len);
        return to_bits_le(vec, num);
    }

    template<typename T>
    __device__  void BasicOper::print_hex(T num) {
        int size = sizeof(T);
        switch (size) {
            case 1: {
                printf("0x%02x", num);break;
            }
            case 2: {
                printf("0x%04x", num);break;
            }
            case 4: {
                printf("0x%08x", num);break;
            }
            case 8: {
                printf("0x%llx", (unsigned  long long) num);break;
            }
            case 16: {
                __uint64_t high = (__uint64_t)(num >> 64);
                __uint64_t low = (__uint64_t)num;
                if (high != 0) {
                    printf("0x%016llx%016llx", (unsigned  long long) high, (unsigned  long long)low);
                } else {
                    printf("0x%llx", (unsigned  long long)low);
                }
                break;
            }
        }
        printf("\n");
    }

     __device__ void acquire_lock(int *lock);
     __device__ void release_lock(int *lock);
}


#endif //CUDA_SNARKVM_BASICOPER_H