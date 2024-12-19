//
// Created by fil on 24-10-16.
//

#include "VmCache.cuh"

 __constant__ int MAX_THREAD=1024;
 __constant__  int cache_type_size = 6;
//tiny=4, smaller=16, small=32, middle=64, large=256, huge=1024
 __constant__  int cache_type_len[6] = {4, 16, 32, 64, 256, 1024}; //对数组来说，存储几种长度的数组
 __constant__  int cache_type_amount[6] = {1, 1, 1, 1, 1, 1}; //每种长度的数组存储几个

__device__ VmCache* cacheInstance = nullptr; //存储单例
__device__ int cache_is_init = 0;

__device__  void VmCache::createBigInt(int threadId){
    int int_size = 1024;
    //128位
    if (nullptr == this->vecBigInt) {
        this->vecBigInt = new HashMap<int,Vec<skm::BigInt*>*>(MAX_THREAD);
    }
    if (nullptr != this->vecBigInt->find(threadId)) {
        return;
    }

    Vec<skm::BigInt*>* vec = Vec<skm::BigInt*>::init(int_size);
    for (int i = 0; i < int_size; i++) {
        vec->push(new skm::BigInt());
    }
    this->vecBigInt->insert(threadId, vec);
}

__device__ void VmCache::createBigIntLst(int threadId) {
    if (nullptr == this->vecBigIntLst) {
        this->vecBigIntLst = new HashMap<int,HashMap<int,Vec<Vec<skm::BigInt *> *> *> *>(MAX_THREAD);
    }

    if (nullptr == this->vecBigIntLst->find(threadId)) {
        this->vecBigIntLst->insert(threadId, this->createLst<Vec<skm::BigInt*>>());
    }
}

__device__ void VmCache::createVecBigIntEmptyLst(int threadId) {
    if (nullptr == this->vecBigIntEmptyLst) {
        this->vecBigIntEmptyLst = new HashMap<int,Vec<Vec<skm::BigInt*>*>*>(MAX_THREAD);
    }

    if (nullptr == this->vecBigIntEmptyLst->find(threadId)) {
        this->vecBigIntEmptyLst->insert(threadId, this->createEmptyLst<Vec<skm::BigInt*>>(10));
    }
}

__device__ void VmCache::createVecU8EmptyLst(int threadId) {
    if (nullptr == this->vecU8EmptyLst) {
        this->vecU8EmptyLst = new HashMap<int,Vec<Vec<u8>*>*>(MAX_THREAD);
    }

    if (nullptr == this->vecU8EmptyLst->find(threadId)) {
        this->vecU8EmptyLst->insert(threadId, this->createEmptyLst<Vec<u8>>(10));
    }
}

__device__  void VmCache::createBoolLst(int threadId){
    if (nullptr == this->vecBoolLst) {
       this->vecBoolLst =  new HashMap<int,HashMap<int,Vec<Vec<bool>*>*>*>(MAX_THREAD);
    }
    if (nullptr == this->vecBoolLst->find(threadId)) {
        this->vecBoolLst->insert(threadId, this->createLst<Vec<bool>>());
    }
}
__device__  void VmCache::createBoolStructLst(int threadId){
    if (nullptr == this->vecBoolStructLst) {
        this->vecBoolStructLst = new HashMap<int,HashMap<int,Vec<Vec<Bool>*>*>*>(MAX_THREAD);
    }
    if (nullptr == this->vecBoolStructLst->find(threadId)) {
        this->vecBoolStructLst->insert(threadId, this->createLst<Vec<Bool>>());
    }
}

__device__  void VmCache::createU8Lst(int threadId) {
    if (nullptr == this->vecU8Lst) {
        this->vecU8Lst = new HashMap<int,HashMap<int,Vec<Vec<u8>*>*>*>(MAX_THREAD);
    }
    if (nullptr == this->vecU8Lst->find(threadId)) {
        this->vecU8Lst->insert(threadId, this->createLst<Vec<u8>>());
    }
}

__device__  void VmCache::createU64Lst(int threadId) {
    if (nullptr == this->vecU64Lst) {
        this->vecU64Lst = new HashMap<int,HashMap<int,Vec<Vec<u64>*>*>*>(MAX_THREAD);
    }
    if (nullptr == this->vecU64Lst->find(threadId)) {
        this->vecU64Lst->insert(threadId, this->createLst<Vec<u64>>());
    }
}

__device__  void VmCache::createBoolStructLstLst(int threadId) {
    if (nullptr == this->vecBoolStructLstLst) {
        this->vecBoolStructLstLst = new HashMap<int,HashMap<int,Vec<Vec<Vec<Bool>*>*>*>*>(MAX_THREAD);
    }
    if (nullptr == this->vecBoolStructLstLst->find(threadId)) {
        this->vecBoolStructLstLst->insert(threadId, this->createLst<Vec<Vec<Bool>*>>());
    }
}

__device__  void VmCache::createBoolLstLst(int threadId) {
    if (nullptr == this->vecBoolLstLst) {
        this->vecBoolLstLst = new HashMap<int,HashMap<int,Vec<Vec<Vec<bool>*>*>*>*>(MAX_THREAD);
    }
    if (nullptr == this->vecBoolLstLst->find(threadId)) {
        this->vecBoolLstLst->insert(threadId, this->createLst<Vec<Vec<bool>*>>());
    }
}

__device__ void VmCache::createU8LstLst(int threadId) {
    if (nullptr == this->vecU8LstLst) {
        this->vecU8LstLst = new HashMap<int,HashMap<int,Vec<Vec<Vec<u8>*>*>*>*>(MAX_THREAD);
    }
    if (nullptr == this->vecU8LstLst->find(threadId)) {
        this->vecU8LstLst->insert(threadId, this->createLst<Vec<Vec<u8>*>>());
    }
}

__device__  void VmCache::createBigIntLstLst(int threadId) {
    if (nullptr == this->vecBigIntLstLst) {
        this->vecBigIntLstLst = new HashMap<int,HashMap<int,Vec<Vec<Vec<skm::BigInt*>*>*>*>*>(MAX_THREAD);
    }
    if (nullptr == this->vecBigIntLstLst->find(threadId)) {
        this->vecBigIntLstLst->insert(threadId, this->createLst<Vec<Vec<skm::BigInt*>*>>());
    }
}
__device__  void VmCache::init()  {
    if (nullptr == cacheInstance) {
        skm::acquire_lock(&cache_is_init);
        if (nullptr == cacheInstance) {
            cacheInstance = new VmCache();
//            __syncthreads();
        }
        skm::release_lock(&cache_is_init);
    }
    VmCache *instance = getInstance();
    skm::acquire_lock(&cache_is_init);
    int threadId = getThreadId();
    instance->createBigInt(threadId);
    instance->createBigIntLst(threadId);
    instance->createBoolLst(threadId);
    instance->createBoolStructLst(threadId);
    instance->createU8Lst(threadId);
    instance->createU64Lst(threadId);
    instance->createBoolLstLst(threadId);
    instance->createBoolStructLstLst(threadId);
    instance->createBigIntLstLst(threadId);
    instance->createVecBigIntEmptyLst(threadId);
    instance->createVecU8EmptyLst(threadId);
    instance->createU8LstLst(threadId);
    skm::release_lock(&cache_is_init);
}

//单例
__device__ VmCache* VmCache::getInstance() {
    return cacheInstance;
}

__device__  int VmCache::getThreadId() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__  int VmCache::getGrade(int len) {
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

//=================获取和归还======================
__device__  void VmCache::returnBigInt(skm::BigInt *val) {
//    printf("归还前: %d\n", getInstance()->vecBigInt->find(getThreadId())->getSize());
    Vec<skm::BigInt*>* vec = getInstance()->vecBigInt->find(getThreadId());
    vec->push(val);
    val->reset(); //置0
//    printf("归还后: %d\n", getInstance()->vecBigInt->find(getThreadId())->getSize());
}

__device__  skm::BigInt* VmCache::getBigInt() {
    Vec<skm::BigInt*>* vec = getInstance()->vecBigInt->find(getThreadId());
    if (vec->getSize() != 0) {
       return vec->pop();
    }
//    printf("剩余: %d\n", getInstance()->vecBigInt->find(getThreadId())->getSize());
    return new skm::BigInt();
}

__device__ Vec<skm::BigInt *> *VmCache::getBigIntLst(int len) {
    Vec<Vec<skm::BigInt*>*>* vec = getInstance()->vecBigIntLst->find(getThreadId())->find(getGrade(len));
    return getList(vec, len);
}

__device__ Vec<skm::BigInt *> *VmCache::getBigIntEmptyLst() {
    Vec<Vec<skm::BigInt*>*>* vec = getInstance()->vecBigIntEmptyLst->find(getThreadId());
    if (vec->getSize() != 0) {
       return vec->pop();
    }
    return Vec<skm::BigInt*>::initEmpty(0);
}

__device__ Vec<u8> *VmCache::getU8EmptyLst() {
    Vec<Vec<u8>*>* vec =getInstance()->vecU8EmptyLst->find(getThreadId());
    if (vec->getSize() != 0) {
        return vec->pop();
    }
    return Vec<u8>::initEmpty(0);
}

__device__ Vec<bool> *VmCache::getBoolLst(int len) {
    Vec<Vec<bool>*>* vec =getInstance()->vecBoolLst->find(getThreadId())->find(getGrade(len));
    return getList(vec, len);
}

__device__ Vec<Bool> *VmCache::getBoolStructLst(int len) {
    Vec<Vec<Bool>*>* vec =getInstance()->vecBoolStructLst->find(getThreadId())->find(getGrade(len));
    return getList(vec, len);
}

__device__ Vec<u8> *VmCache::getU8Lst(int len) {
    Vec<Vec<u8>*>* vec =getInstance()->vecU8Lst->find(getThreadId())->find(getGrade(len));
    Vec<u8>* rst = getList(vec, len);
    for (int i = 0; i < rst->getCapacity(); i++) { //置0
       rst->set(i, 0);
    }
    return rst;
}

__device__ Vec<u64> *VmCache::getU64Lst(int len) {
    Vec<Vec<u64>*>* vec =getInstance()->vecU64Lst->find(getThreadId())->find(getGrade(len));
    Vec<u64>* rst = getList(vec, len);
    for (int i = 0; i < rst->getCapacity(); i++) {//置0
        rst->set(i, 0);
    }
    return rst;
}

__device__ Vec<Vec<Bool>*> *VmCache::getBoolStructLstLst(int len) {
    Vec<Vec<Vec<Bool>*>*>* vec =getInstance()->vecBoolStructLstLst->find(getThreadId())->find(getGrade(len));
    return getList(vec, len);
}

__device__ Vec<Vec<bool>*> *VmCache::getBoolLstLst(int len) {
    Vec<Vec<Vec<bool>*>*>* vec =getInstance()->vecBoolLstLst->find(getThreadId())->find(getGrade(len));
    return getList(vec, len);
}

__device__ Vec<Vec<u8> *> *VmCache::getU8LstLst(int len) {
    Vec<Vec<Vec<u8>*>*>* vec =getInstance()->vecU8LstLst->find(getThreadId())->find(getGrade(len));
    return getList(vec, len);
}

__device__ Vec<Vec<skm::BigInt*>*> *VmCache::getBigIntLstLst(int len) {
    Vec<Vec<Vec<skm::BigInt*>*>*>* vec =getInstance()->vecBigIntLstLst->find(getThreadId())->find(getGrade(len));
    return getList(vec, len);
}

__device__ void VmCache::returnBigIntLst(Vec<skm::BigInt *> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    getInstance();
    Vec<Vec<skm::BigInt*>*>* array = getInstance()->vecBigIntLst->find(getThreadId())->find(getGrade(vec->getCapacity()));

    array->push(vec);
    for (int i = 0; i < len; i++) {
       skm::BigIntOper::destory(vec->get(i));
    }
    for (int i = 0; i < vec->getCapacity(); i++) { //置空
        vec->set(i, nullptr);
    }
}

__device__ void VmCache::returnBigIntLstOnly(Vec<skm::BigInt *> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<skm::BigInt*>*>* array = getInstance()->vecBigIntLst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);
    for (int i = 0; i < vec->getCapacity(); i++) { //置空
        vec->set(i, nullptr);
    }
}

__device__ void VmCache::returnBoolLst(Vec<bool> *vec) {
    if (nullptr == vec) {
        return;
    }
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<bool>*>* array = getInstance()->vecBoolLst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);
//    for (int i = 0; i < vec->getCapacity(); i++) { //置0
//        vec->set(i, false);
//    }
}

__device__ void VmCache::returnBoolStructLst(Vec<Bool> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<Bool>*>* array = getInstance()->vecBoolStructLst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);
}

__device__ void VmCache::returnU8Lst(Vec<u8> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<u8>*>* array = getInstance()->vecU8Lst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);
}

__device__ void VmCache::returnU64Lst(Vec<u64> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<u64>*>* array = getInstance()->vecU64Lst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);
}

__device__ void VmCache::returnBoolStructLstLst(Vec<Vec<Bool>*> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<Vec<Bool>*>*>* array = getInstance()->vecBoolStructLstLst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);

    for (int i = 0; i < len; i++) {
        returnBoolStructLst(vec->get(i));
    }
    for (int i = 0; i < vec->getCapacity(); i++) { //置空
        vec->set(i, nullptr);
    }
}

__device__ void VmCache::returnBoolLstLst(Vec<Vec<bool>*> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<Vec<bool>*>*>* array = getInstance()->vecBoolLstLst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);
    for (int i = 0; i < len; i++) {
        returnBoolLst(vec->get(i));
    }
    for (int i = 0; i < vec->getCapacity(); i++) { //置空
        vec->set(i, nullptr);
    }
}

__device__ void VmCache::returnU8LstLst(Vec<Vec<u8> *> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<Vec<u8>*>*>* array = getInstance()->vecU8LstLst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);
    for (int i = 0; i < len; i++) {
        returnU8Lst(vec->get(i));
    }
    for (int i = 0; i < vec->getCapacity(); i++) { //置空
        vec->set(i, nullptr);
    }
}


__device__ void VmCache::returnBigIntLstLst(Vec<Vec<skm::BigInt*>*> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<Vec<skm::BigInt*>*>*>* array = getInstance()->vecBigIntLstLst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);

    for (int i = 0; i < len; i++) {
        returnBigIntLst(vec->get(i));
    }
    for (int i = 0; i < vec->getCapacity(); i++) { //置空
        vec->set(i, nullptr);
    }
}

__device__ void VmCache::returnBigIntLstLstOnly(Vec<Vec<skm::BigInt *> *> *vec) {
    int len = vec->getSize();
    vec->reset_size();
    Vec<Vec<Vec<skm::BigInt*>*>*>* array = getInstance()->vecBigIntLstLst->find(getThreadId())->find(getGrade(vec->getCapacity()));
    array->push(vec);

    for (int i = 0; i < len; i++) {
        returnBigIntLstOnly(vec->get(i));
    }
    for (int i = 0; i < vec->getCapacity(); i++) { //置空
       vec->set(i, nullptr);
    }
}

__device__ void VmCache::returnBigIntEmptyLstOnly(Vec<skm::BigInt *> *vec) {
    Vec<Vec<skm::BigInt*>*>* array = getInstance()->vecBigIntEmptyLst->find(getThreadId());
    array->push(vec);
    vec->resetEmpty(); //重置
}

__device__ void VmCache::returnU8EmptyLstOnly(Vec<u8> *vec) {
    Vec<Vec<u8>*>* array = getInstance()->vecU8EmptyLst->find(getThreadId());
    array->push(vec);
    vec->resetEmpty(); //重置
}






