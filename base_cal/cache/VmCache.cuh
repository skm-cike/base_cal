//
// Created by fil on 24-10-16.
//

#ifndef SNARKVM_CUDA_VMCACHE_CUH
#define SNARKVM_CUDA_VMCACHE_CUH
#include "../base_vec.cuh"
#include "../base_map.cuh"
#include "../bigint.cuh"
#include <thrust/device_vector.h>
#include <thrust/pair.h>

 __constant__ extern int MAX_THREAD;

 __constant__ extern int cache_type_size;
//tiny=4, smaller=16, small=32, middle=64, large=256, huge=1024
 __constant__ extern int cache_type_len[6]; //对数组来说，存储几种长度的数组
 __constant__ extern int cache_type_amount[6]; //每种长度的数组存储几个

class VmCache {
private:
    __device__ VmCache(){}; //构造函数
    HashMap<int,Vec<skm::BigInt*>*>* vecBigInt = nullptr;        //存储BigInt
    HashMap<int,HashMap<int,Vec<Vec<skm::BigInt*>*>*>*>* vecBigIntLst = nullptr;//存储BigInt数组
    HashMap<int,HashMap<int,Vec<Vec<bool>*>*>*>* vecBoolLst = nullptr;             //存储bool数组
    HashMap<int,HashMap<int,Vec<Vec<Bool>*>*>*>* vecBoolStructLst = nullptr;            //存储Bool数组
    HashMap<int,HashMap<int,Vec<Vec<u8>*>*>*>* vecU8Lst = nullptr;                    //存储u8的数组
    HashMap<int,HashMap<int,Vec<Vec<u64>*>*>*>* vecU64Lst = nullptr;                    //存储u8的数组
    HashMap<int,HashMap<int,Vec<Vec<Vec<Bool>*>*>*>*>* vecBoolStructLstLst = nullptr;            //存储Bool*空数组数组
    HashMap<int,HashMap<int,Vec<Vec<Vec<bool>*>*>*>*>* vecBoolLstLst = nullptr;            //存储bool*空数组数组
    HashMap<int,HashMap<int,Vec<Vec<Vec<u8>*>*>*>*>* vecU8LstLst = nullptr;            //存储u8*空数组数组
    HashMap<int,HashMap<int,Vec<Vec<Vec<skm::BigInt*>*>*>*>*>* vecBigIntLstLst = nullptr;            //存储BigInt*空数组数组
    HashMap<int,Vec<Vec<skm::BigInt*>*>*>* vecBigIntEmptyLst = nullptr;//存储BigInt空数组，数组未初始化
    HashMap<int,Vec<Vec<u8>*>*>* vecU8EmptyLst = nullptr;

    template<typename T>
    __device__  HashMap<int, Vec<T*>*>* createLst();
    template<typename T>
    __device__  Vec<T*>* createEmptyLst(int amount);
    __device__  void createBigIntLst(int threadId);
    __device__  void createBigInt(int threadId);
    __device__  void createBoolLst(int threadId);
    __device__  void createBoolStructLst(int threadId);
    __device__  void createU8Lst(int threadId);
    __device__  void createU64Lst(int threadId);
    __device__  void createBoolStructLstLst(int threadId);
    __device__  void createBoolLstLst(int threadId);
    __device__  void createU8LstLst(int threadId);
    __device__  void createBigIntLstLst(int threadId);
    __device__  void createVecBigIntEmptyLst(int threadId);
    __device__  void createVecU8EmptyLst(int threadId);

    template<typename T>
    __device__  static T* getList(Vec<T*>* vec, int len);
    __device__  static VmCache* getInstance();
    __device__  static int getGrade(int len);
public:
    // 防止拷贝和赋值
    __device__ VmCache(const VmCache&) = delete;
    __device__ VmCache& operator=(const VmCache&) = delete;
    //初始化数据
    __device__ static void init();
    __device__ static int getThreadId();
    __device__ static skm::BigInt* getBigInt();
    __device__ static void returnBigInt(skm::BigInt* val);
    //获取和归还
    __device__ static Vec<skm::BigInt*>* getBigIntLst(int len);
    __device__ static Vec<bool>*  getBoolLst(int len);
    __device__ static Vec<Bool>*  getBoolStructLst(int len);
    __device__ static Vec<u8>*  getU8Lst(int len);
    __device__ static Vec<u64>*  getU64Lst(int len);
    __device__ static Vec<Vec<Bool>*>* getBoolStructLstLst(int len);
    __device__ static Vec<Vec<bool>*>* getBoolLstLst(int len);
    __device__ static Vec<Vec<u8>*>* getU8LstLst(int len);
    __device__ static Vec<Vec<skm::BigInt*>*>* getBigIntLstLst(int len);
    __device__ static Vec<skm::BigInt*>* getBigIntEmptyLst();
    __device__ static Vec<u8>*  getU8EmptyLst();

    __device__ static void returnBigIntLst(Vec<skm::BigInt*>* vec);
    __device__ static void returnBigIntLstOnly(Vec<skm::BigInt*>* vec);
    __device__ static void returnBoolLst(Vec<bool>* vec);
    __device__ static void returnBoolStructLst(Vec<Bool>* vec);
    __device__ static void returnU8Lst(Vec<u8>* vec);
    __device__ static void returnU64Lst(Vec<u64>* vec);
    __device__ static void returnBoolStructLstLst(Vec<Vec<Bool>*>* vec);
    __device__ static void returnBoolLstLst(Vec<Vec<bool>*>* vec);
    __device__ static void returnU8LstLst(Vec<Vec<u8>*>* vec);
    __device__ static void returnBigIntLstLst(Vec<Vec<skm::BigInt*>*>* vec);
    __device__ static void returnBigIntLstLstOnly(Vec<Vec<skm::BigInt*>*>* vec);
    __device__ static void returnBigIntEmptyLstOnly(Vec<skm::BigInt*>* vec);
    __device__ static void returnU8EmptyLstOnly(Vec<u8>* vec);
};

template<typename T>
__device__ HashMap<int, Vec<T*>*>* VmCache::createLst() {
    HashMap<int, Vec<T*>*>* type = new HashMap<int, Vec<T*>*>(cache_type_size);
    for (int i = 0; i < cache_type_size; i++) {
        Vec<T*>* vec = Vec<T*>::init(cache_type_amount[i]);
        for (int j = 0; j < cache_type_amount[i]; j++) {
            T* newVec = T::init(cache_type_len[i]);
            vec->push(newVec);
        }
        type->insert(i, vec);
    }
    return type;
}

template<typename T>
__device__ Vec<T*>* VmCache::createEmptyLst(int amount) {
    Vec<T*>* vec = Vec<T*>::init(amount);
    for (int j = 0; j < amount; j++) {
        vec->push(T::initEmpty(4096));
    }
    return vec;
}

template<typename T>
__device__ T* VmCache::getList(Vec<T*>* vec, int len) {
    if (vec->getSize() != 0) { //有值，从缓存里面取
        T* r = vec->pop();
       return r;
    }
    T* r = T::init(cache_type_len[getGrade(len)]); //没有值，则新建一个
    return r;
}


#endif //SNARKVM_CUDA_VMCACHE_CUH
