//
// Created by fil on 24-10-23.
//
#include "base_type.cuh"
//====================================普通常量=================================================
__device__ __constant__  u8 MAX_U8=0xff;
__device__ __constant__  u16 MAX_U16=0xffff;
__device__ __constant__  u32 MAX_U32=0xffffffff;
__device__ __constant__  u64 MAX_U64=0xffffffffffffffff;
__device__ __constant__  i8 MAX_I8=0x7f;
__device__ __constant__  i16 MAX_I16=0x7fff;
__device__ __constant__  i32 MAX_I32=0x7fffffff;
__device__ __constant__  i64 MAX_I64=0x7fffffffffffffff;
__device__ __constant__  i8 MIN_I8=0x80;
__device__ __constant__  i16 MIN_I16=0x8000;
__device__ __constant__  i32 MIN_I32=0x80000000;
__device__ __constant__  i64 MIN_I64=0x8000000000000000;
__device__ __constant__  U128 MAX_U128=0xffffffffffffffffffffffffffffffff;
__device__ __constant__  I128 MAX_I128=0x7fffffffffffffffffffffffffffffff;
__device__ __constant__  I128 MIN_I128=0x80000000000000000000000000000000;

