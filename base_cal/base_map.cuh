//
// Created by fil on 24-10-16.
//

#ifndef SNARKVM_CUDA_BASE_MAP_CUH
#define SNARKVM_CUDA_BASE_MAP_CUH
class SimpleHash {
public:
    // CUDA 简单哈希函数 (djb2)
    __device__  static unsigned long hash_djb2(const unsigned char* data, unsigned long length) {
        unsigned long hash = 5381;
        for (int i = 0; i < length; ++i) {
            hash = ((hash << 5) + hash) + data[i]; // hash * 33 + data[i]
        }
        return hash;
    }

// 任意类型的哈希计算
    template <typename T>
    __device__ static unsigned long hash_any(T value) {
        // 将任意类型转换为字节数组
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&value);
        unsigned long length = sizeof(T);
        return hash_djb2(bytes, length);
    }
};

// 哈希表结构
template<typename K, typename T>
class HashMap {
private:
    int size;
    HashItem<K,T>* items;
    // 哈希函数
    __device__ unsigned long hash(K key);
public:
    __device__ HashMap(int size);
    __device__ HashMap();
    // 插入键值对
    __device__ bool insert(K key, T value);
    // 查找键值对
    __device__ T find(K key);
};

template<typename K, typename T>
__device__ HashMap<K, T>::HashMap(int size) {
    this->items = new HashItem<K,T>[size];
    this->size = size;
    for (int i = 0; i < size; i++) {
        this->items[i].occupied = false; // 初始化为未占用
    }
}

template<typename K, typename T>
__device__ HashMap<K, T>::HashMap() {
    HashMap<K,T>(1024);
}

// 哈希函数
template<typename K, typename T>
__device__ unsigned long HashMap<K, T>::hash(K key) {
    return key % size; // 简单的模运算
//    return SimpleHash::hash_any(key) % size;
}

// 插入键值对
template<typename K, typename T>
__device__ bool HashMap<K, T>::insert(K key, T value) {
    unsigned long idx = hash(key);
    unsigned long startIdx = idx; // 记录开始索引以防循环
    do {
        if (!items[idx].occupied) {
            items[idx].key = key;
            items[idx].value = value;
            items[idx].occupied = true;
            return true;
        }
        // 线性探测下一个索引
        idx = (idx + 1) % size;
    } while (idx != startIdx); // 避免死循环
    return false;
}

// 查找键值对
template<typename K, typename T>
__device__ T HashMap<K, T>::find(K key) {
    unsigned long idx = hash(key);
    unsigned long startIdx = idx; // 记录开始索引以防循环
    do {
        if (items[idx].occupied && items[idx].key == key) {
            return items[idx].value; // 找到键，返回值
        }
        idx = (idx + 1) % size; // 线性探测
    } while (idx != startIdx && items[idx].occupied); // 如果没有找到，继续查找
    printf("没有找到数据:\n");
    return (T)(nullptr); // 如果未找到，返回-1
}

#endif //SNARKVM_CUDA_BASE_MAP_CUH
