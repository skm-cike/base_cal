//
// Created by fil on 24-9-19.
//

#ifndef SNARKVM_CUDA_BASE_VEC_CUH
#define SNARKVM_CUDA_BASE_VEC_CUH
#include "stdio.h"
#include "base_type.cuh"
constexpr U64 INIT_CAPACITY = 32;
template<typename T>
class Vec {
private:
    VecStruct<T> entity;
    __device__ void resize(); //扩容
public:
    __device__  void destory_unit();
    __device__ void destory_unit(int start, int len);
    __device__  void destory();
    __device__  void destoryEmpty();
    __device__  void resetEmpty(); //重置为空，数组为空指针
    __device__  Vec<T>* sub_array_change(int start_index, int length);
    __device__  Vec<T>* sub_array_change(int start_index);
    __device__  Vec<T>* sub_array_change(Vec<T>* rst, int start_index, int length);
    __device__  Vec<T>* sub_array_change(Vec<T>* rst, int start_index);
    __device__  Vec<T>* sub_array(int start_index, int length);
    __device__  Vec<T>* sub_array(int start_index);
    __device__  static Vec<T>* init(int capacity);
    __device__  static Vec<T>* init();
    __device__  static Vec<T>* initEmpty(int capacity);
    //初始化并填充值
    __device__  static Vec<T>* init(int capacity, T val);
    __device__  Vec<Vec<T> *> *chunks(int chunk_size);
    __device__ int getCapacity();
    __device__ int getSize();
    __device__ void setSize(int size);
    // 设置size为最大大小,以完全使用数组
    __device__ void toMaxSize();
    __device__ T get(int index);
    __device__ void set(int index, T a);
    __device__  Vec<T>* clone();

    __device__  void push(T* val, int size);
    //填充并修改值
    __device__ void fill(Vec<T>* other);
    __device__  void push(T val);
    __device__  void insert(int start_index, T newVal);
    __device__  void insert(int start_index, Vec<T>* newVal);

    __device__  T pop();
    __device__  T swap_remove(int index);
    __device__  void print_bool();
    __device__  void print_int();
    __device__  void print_longint();
    __device__  void reset_size();


    __device__  Vec<T>* reverse();
    __device__ void copy_from_slice(const Vec<T> *src, int start, int len);
    __device__ void copy_from_slice(const Vec<T> *src);

    __device__ Vec<Vec<T>*> * split_at(int num);
};


//=============================================实现===============================================================

template<typename T>
__device__ void Vec<T>::resize() {
    int new_capacity = this->entity.capacity * 3 / 2 + 1;
    T* newVal = new T[new_capacity];
    for (int i = 0; i < this->entity.size; i++) {
        newVal[i] = this->entity.val[i];
    }
    delete(this->entity.val);
    this->entity.val = newVal;
    this->entity.capacity = new_capacity;
//    printf("--------------新数组长度: %d \n",new_capacity);
}

template<typename T>
__device__ int Vec<T>::getCapacity() {
    return this->entity.capacity;
}

template<typename T>
__device__ int Vec<T>::getSize() {
    return this->entity.size;
}

template<typename T>
__device__ void Vec<T>::setSize(int size) {
    this->entity.size = size;
}

// 设置size为最大大小,以完全使用数组
template<typename T>
__device__ void Vec<T>::toMaxSize() {
    this->entity.size = this->entity.capacity;
}

template<typename T>
__device__ T Vec<T>::get(int index) {
    return this->entity.val[index];
}

template<typename T>
__device__ void Vec<T>::set(int index, T a) {
    this->entity.val[index] = a;
}

template<typename T>
__device__  Vec<T>* Vec<T>::clone() {
    Vec<T>* newVec = init(this->entity.capacity);
    newVec->setSize(this->getSize());
    for (int i = 0; i < this->entity.size; i++) {
        newVec->entity.val[i] = this->entity.val[i];
    }
    return newVec;
}

//初始化一个空数组
template<typename T>
__device__ Vec<T> *Vec<T>::initEmpty(int capacity) {
    Vec<T>* vec = new Vec<T>();
    vec->entity.size = 0;
    vec->entity.capacity = capacity;
    vec->entity.val = nullptr;
    return vec;
}

template<typename T>
__device__  Vec<T>* Vec<T>::init(int capacity) {
//    printf("--------------数组长度: %d \n",capacity);
    Vec<T>* vec = new Vec<T>();
    vec->entity.size = 0;
    vec->entity.capacity = capacity;
    vec->entity.val = new T[vec->entity.capacity];
    return vec;
}

template<typename T>
__device__   Vec<T>* Vec<T>::init() {
    return init(INIT_CAPACITY);
}

//初始化并填充值
template<typename T>
__device__   Vec<T>* Vec<T>::init(int capacity, T val) {
//    printf("--------------数组长度: %d \n",capacity);
    Vec<T>* vec = init(capacity);
    for (int i = 0; i < capacity; i++) {
        vec->push(val);
    }
    return vec;
}

template<typename T>
__device__  void Vec<T>::push(T* val, int size) {
    int newSize = this->entity.size + size;
    while (newSize > this->entity.capacity) {
        resize();
    }
    for (int i = 0; i < size; i++) {
        this->entity.val[this->entity.size + i] = val[i];
    }
    this->entity.size = newSize;
}

//填充并修改值
template<typename T>
__device__ void Vec<T>::fill(Vec<T>* other) {
    for (int i = 0; i < other->getSize(), i < this->getSize(); i++) {
        this->set(i, other->get(i));
    }
}

template<typename T>
__device__  void Vec<T>::push(T val) {
    int newSize = this->entity.size + 1;
    while (newSize > this->entity.capacity) {
        resize();
    }
    this->entity.val[this->entity.size] = val;
    this->entity.size = newSize;
}

template<typename T>
__device__  void Vec<T>::insert(int start_index, T newVal) {
    int newSize = this->entity.size + 1;
    while (newSize > this->entity.capacity) {
        resize();
    }
    this->entity.size = newSize;
    //移动原数组
    for (int i = this->entity.size-1; i > start_index; i--) {
        this->entity.val[i] = this->entity.val[i - 1];
    }
    // 插入新数据
    this->entity.val[start_index] = newVal;
}

template<typename T>
__device__  void Vec<T>::insert(int start_index, Vec<T>* newVal) {
    int newSize = this->entity.size + newVal->entity.size;
    while (newSize > this->entity.capacity) {
        resize();
    }
    this->entity.size = newSize;
    //移动原数组
    for (int i = this->entity.size-1; i > start_index; i--) {
        this->entity.val[i + newVal->entity.size - 1] = this->entity.val[i - 1];
    }
    // 插入新数组的元素
    for (int i = 0; i < newVal->entity.size; i++) {
        this->entity.val[start_index + i] = newVal->entity.val[i];
    }
}

template<typename T>
__device__  Vec<T>* Vec<T>::sub_array(int start_index, int length) {
    int end_index = start_index + length;
    Vec<T>* rst = init(length);
    for (int i = start_index; i < end_index, i < this->entity.size; i++) {
        rst->push(this->entity.val[i]);
    }
    return rst;
}

template<typename T>
__device__  Vec<T>* Vec<T>::sub_array(int start_index) {
    int length = this->entity.size - start_index;
    return sub_array(start_index, length);
}

template<typename T>
__device__  Vec<T>* Vec<T>::sub_array_change(int start_index, int length) {
    Vec<T>* rst = new Vec<T>();
    return sub_array_change(rst, start_index, length);
}

template<typename T>
__device__  Vec<T>* Vec<T>::sub_array_change(int start_index) {
    int length = this->entity.size - start_index;
    return sub_array_change(start_index, length);
}

template<typename T>
__device__  Vec<T>* Vec<T>::sub_array_change(Vec<T>* rst, int start_index, int length) {
    rst->entity.capacity = length;
    rst->entity.size = length;
    rst->entity.val = &this->entity.val[start_index];
    return rst;
}

template<typename T>
__device__  Vec<T>* Vec<T>::sub_array_change(Vec<T>* rst, int start_index) {
    int length = this->entity.size - start_index;
    return sub_array_change(rst, start_index, length);
}

template<typename T>
__device__  T Vec<T>::pop() {
    int index = this->entity.size-1;
    T val = this->entity.val[index];
    this->entity.size--;
    return val;
}

template<typename T>
__device__  T Vec<T>::swap_remove(int index) {
    T rst = this->entity.val[index];
    this->entity.val[index] = this->entity.val[this->entity.size-1];
    this->entity.size--;
    return rst;
}

template<typename T>
__device__  void Vec<T>::print_bool() {
    printf("[");
    for (int i=0; i < this->entity.size; i++) {
        if (this->entity.val[i]) {
            printf("true");
        } else {
            printf("false");
        }
        if (i < this->entity.size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}
template<typename T>
__device__  void Vec<T>::print_int() {
    printf("[");
    for (int i=0; i < this->entity.size; i++) {
        printf("%d", this->entity.get(i));
        if (i < this->entity.size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

template<typename T>
__device__  void Vec<T>::print_longint() {
    printf("[");
    for (int i=0; i < this->entity.size; i++) {
        printf("%lld", this->entity.get(i));
        if (i < this->entity.size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}
template<typename T>
__device__  void Vec<T>::reset_size() {
    this->entity.size = 0;
}
template<typename T>
__device__  void Vec<T>::destory_unit() {
    destory_unit(0, this->getSize());
}
template<typename T>
__device__ void Vec<T>::destory_unit(int start, int len) {
    for (int i = start; i < len; i++) {
        if (isPointer(this->entity.val[i])) {
            delete (this->entity.val[i]);
        }
    }
}
template<typename T>
__device__  void Vec<T>::destory() {
    delete(this->entity.val);
    delete(this);
}

template<typename T>
__device__ void Vec<T>::destoryEmpty() {
    this->entity.val = nullptr;
    delete(this);
}

template<typename T>
__device__ void Vec<T>::resetEmpty() { //重置为空指针
    this->entity.val = nullptr;
    this->setSize(0);
}

template<typename T>
__device__  Vec<T>* Vec<T>::reverse() {
    int start = 0;            // 开始索引
    int end = this->entity.size - 1;      // 结束索引
    T temp;                // 临时变量用于交换
    while (start < end) {
        // 交换元素
        temp = this->entity.val[start];
        this->entity.val[start] = this->entity.val[end];
        this->entity.val[end] = temp;
        // 移动索引
        start++;
        end--;
    }
    return this;
}

template<typename T>
__device__ void Vec<T>::copy_from_slice(const Vec<T> *src, int start, int len) {
    for (int i=0; i < len; i++) {
        this->entity.val[i+start] = src->entity.val[i];
    }
}
template<typename T>
__device__ void Vec<T>::copy_from_slice(const Vec<T> *src) {
    copy_from_slice(src, 0, src->entity.size);
}

template<typename T>
__device__  Vec<Vec<T> *> *Vec<T>::chunks(int chunk_size) {
    const int numChunks = (this->getSize() + chunk_size - 1) / chunk_size; // 计算块的数量
    Vec<Vec<T> *> *newVecs = Vec<Vec<T> *>::init(numChunks);
    Vec<T> *unit = nullptr;
    for (int i = 0; i < this->getSize(); i++) {
        if (i % chunk_size == 0) {
            unit = Vec<T>::init(chunk_size);
            newVecs->push(unit);
        }
        unit->push(this->get(i));
    }
    return newVecs;
}

template<typename T>
__device__ Vec<Vec<T>*> * Vec<T>::split_at(int num) {
    Vec<Vec<T>*>* rst = Vec<Vec<T>*>::init(2);
    Vec<T>* r1 = Vec<T>::init(num);
    Vec<T>* r2 = Vec<T>::init(this->getSize() - num);
    for (int i = 0; i < this->getSize(); i++) {
        if (i < num) {
            r1->push(this->get(i));
        } else {
            r2->push(this->get(i));
        }
    }
    rst->push(r1);
    rst->push(r2);
    return rst;
}

#endif //SNARKVM_CUDA_BASE_VEC_CUH
