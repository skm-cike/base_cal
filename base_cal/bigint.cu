//
// Created by fil on 24-10-16.
//
#include "bigint.cuh"
#include "stdio.h"
#include "cache/VmCache.cuh"
namespace skm {
// 函数：比较两个 128 位无符号整数
    __device__  bool BigIntOper::compare_uint128(const BigInt *a, const BigInt *b) {
        if (a->number[1] != b->number[1])
            return a->number[1] > b->number[1];
        return a->number[0] > b->number[0];
    }
// 函数：减去两个 128 位无符号整数
    __device__  BigInt *BigIntOper::subtract_uint128(const BigInt *a, const BigInt *b) {
        BigInt *result = zero(2, false);
        result->number[0] = a->number[0] - b->number[0];
        result->number[1] = a->number[1] - b->number[1] - (a->number[0] < b->number[0]);
        result->is_signed = a->is_signed;
        return result;
    }

    __device__  BigInt* BigIntOper::zero(int length, bool is_signed) {
        return zero(length, is_signed, Mode::Private);
    }
    __device__  BigInt* BigIntOper::zero(int length, bool is_signed, Mode mod) {
        switch (length) {
            case 2: {
                return CreatInt128((__uint64_t)0, (__uint64_t)0, is_signed, mod);
            }
            case 4: {
                return CreatInt256((__uint64_t)0, (__uint64_t)0, (__uint64_t)0, (__uint64_t)0,is_signed, mod);
            }
            case 6: {
                //return CreatInt384((__uint64_t)0, (__uint64_t)0, (__uint64_t)0, (__uint64_t)0, (__uint64_t)0, (__uint64_t)0, is_signed, mod);
            }
            case 8: {
                //return CreatInt512((__uint64_t)0, (__uint64_t)0, (__uint64_t)0, (__uint64_t)0, (__uint64_t)0, (__uint64_t)0, (__uint64_t)0, (__uint64_t)0, is_signed, mod);
            }
        }
    }

    __device__  bool BigIntOper::is_zero(const BigInt* a) {
        for (int i = 0; i < a->size; i++) {
            if (a->number[i] != 0) {
                return false;
            }
        }
        return true;
    }

    __device__  void BigIntOper::to_zero(BigInt* a) {
        for (int i = 0; i < a->size; i++) {
            a->number[i] = 0;
        }
    }

    __device__  bool BigIntOper::isNegative(const BigInt* a) {
        return (a->number[a->size - 1] & (1ULL << 63)) != 0;
    }

    __device__  void BigIntOper::setNegative(BigInt& a, bool negative) {
        if (negative) {
            a.number[a.size - 1] |= (1ULL << 63); //设置最高位位1表示负数
        } else {
            a.number[a.size - 1] &= ~(1ULL << 63); //取消最高位的符号位
        }
    }

    //将补码转为原码
    __device__  BigInt* BigIntOper::to_original(BigInt* a)  {
        for (int i = 0; i < a->size; ++i) {
            a->number[i] = ~a->number[i];
        }
        //补码要+1
        U64 carry =1;
        for (int i = 0; i < a->size; ++i) {
            U64 temp = a->number[i] + carry;
            carry = (temp < a->number[i]) ? 1: 0; //检查进位
            a->number[i] = temp;
            if (0 == carry) break; //无进位则退出
        }
        return a;
    }

    __device__  void BigIntOper::insert(BigInt* a, int begin, int len, __uint64_t val) { //a的长度需要足够长
        int new_size = (*a).size + len;
        (*a).size = new_size;
        for (int index = new_size - 1; index > begin ; --index) {
            (*a).number[index] = (*a).number[index - len];
        }
        for (int index = begin; index < len+begin; ++index) {
            (*a).number[index] = val;
        }
    }

    __device__  BigInt* BigIntOper::shiftLeft(BigInt* a, __uint32_t shift) { //左移，并扩大原数
        if (0 == shift) {
            return a;
        }
        __uint32_t blockShift = shift / 64;
        __uint32_t bitShift = shift % 64;
        __uint64_t  carry = 0;
        int extends = blockShift;
        if (blockShift % 2 == 1) {
            extends++;
        }

        BigInt* result = zero((*a).size + extends + 2, (*a).is_signed); //创建更大的整形 可以存储更多的临时值
        (*result).size = (*a).size;                                            //调整长度

        for (int i = 0; i < (*a).size; ++i) {
            __uint64_t newCarry = (*a).number[i] >> (64 - bitShift);
            (*result).number[i] = ((*a).number[i] << bitShift) | carry;
            carry = newCarry;
        }
        if (carry) {
            (*result).number[(*a).size] = carry;
            (*result).size=(*a).size + 1;
        }

        insert(result,0, blockShift, 0);
        return result;
    }

    __device__  BigInt* BigIntOper::mod(const BigInt* a, const BigInt* mod) {
        BigInt* result = clone(a);
        while (gt(result, mod)) {
            BigInt* shiftedMod = clone(mod);
            int shift = result->size - mod->size;
            shiftedMod = shiftLeft(shiftedMod, shift * 64);
            if (lt(result, shiftedMod)) {
                shiftedMod = shiftLeft(shiftedMod, -64);
                shift--;
            }
            result = sub_w(result, shiftedMod);
        }
        return result;
    }

    // 大整数除法运算，获取余数用于十进制转换
    __device__  U64 BigIntOper::divmod_bigint(BigInt *a, U64 divisor) {
        U64 remainder = 0;
        for (int i = a->size - 1; i >= 0; --i) {
            U128 current = ((U128)remainder << 64) | a->number[i];
            a->number[i] = current / divisor;
            remainder = current % divisor;
        }

        // 去掉多余的高位0
        while (a->size > 0 && a->number[a->size - 1] == 0) {
            a->size--;
        }

        return remainder;
    }

    __device__  U8 BigIntOper::_addcarry_u64(const U8 carry_in, const U64 src1 , const U64 src2, U64 *dst) {
        U64 temp = src1 + src2;
        U64 result = temp + carry_in;
        *dst = result;
        return (temp < src1) || (result < temp);
    }



    __device__  U8 BigIntOper::_subborrow_u64(const U8 borrow_in, const U64 src1, const U64 src2, U64 *dst) {
        U64 temp = src1 - src2;
        U64 result = temp - borrow_in;
        *dst = result;
        return (src1 < src2) || (temp < borrow_in);
    }

    __device__  BigInt* BigIntOper::sub_w(const BigInt* a, const BigInt* b, BigInt* result) {
        int borrow = 0;
        for (int i = 0; i < a->size; ++i) {
            __uint64_t lhs = a->number[i];
            __uint64_t rhs = (i < b->size) ? b->number[i]: 0;
            __uint64_t diff = lhs - rhs - borrow;
            if (lhs < rhs + borrow) {
                borrow = 1;
            } else {
                borrow = 0;
            }
            result->number[i] = diff;
        }
        return result;
    }

    __device__  void BigIntOper::copy_to(const BigInt* a, BigInt* b) {
        for (int i = 0; i < a->size; i++) {
            b->number[i] = a->number[i];
        }
        b->is_signed = a->is_signed;
        b->mod = a->mod;
        b->size = a->size;
    }

    __device__  BigInt* BigIntOper::clone(const BigInt* a) {
        BigInt* result = zero(a->size, a->is_signed);
        copy_to(a, result);
        return result;
    }
    //存储按照低位在前，高位在后的顺序存储,方便编程;  函数的参数则高位在前，方便记忆
    __device__  BigInt* BigIntOper::CreatInt128(__uint64_t num1, __uint64_t num2, bool is_signed) { //128位
        return CreatInt128(num1, num2, is_signed, Mode::Private);
    }
    __device__  BigInt* BigIntOper::CreatInt128(__uint64_t num1, __uint64_t num2, bool is_signed, Mode mod)  { //128位
        int size = 2;
        BigInt* rst = VmCache::getBigInt();
        rst->mod = mod;
        rst->is_signed = is_signed;
        rst->size = size;
        rst->number[0] = num1;
        rst->number[1] = num2;
        return rst;
    }
    __device__  BigInt* BigIntOper::CreatInt256(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, bool is_signed) { //256位
        return CreatInt256(num1, num2, num3, num4, is_signed, Mode::Private);
    }
    __device__  BigInt* BigIntOper::CreatInt256(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, bool is_signed, Mode mod)  { //256位
        int size = 4;
        BigInt* rst = VmCache::getBigInt();
        rst->mod = mod;
        rst->is_signed = is_signed;
        rst->size = size;
        rst->number[0] = num1;
        rst->number[1] = num2;
        rst->number[2] = num3;
        rst->number[3] = num4;
        return rst;
    }
/*
    __device__  BigInt* BigIntOper::CreatInt384(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, __uint64_t num5, __uint64_t num6, bool is_signed) {
        return CreatInt384(num1, num2, num3, num4, num5, num6, is_signed, Mode::Private);
    }
    __device__  BigInt* BigIntOper::CreatInt384(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, __uint64_t num5, __uint64_t num6, bool is_signed, Mode mod)  { //256位
        int size = 6;
        BigInt* rst = VmCache::getBigInt();
        rst->mod = mod;
        rst->is_signed = is_signed;
        rst->size = size;
        rst->number[0] = num1;
        rst->number[1] = num2;
        rst->number[2] = num3;
        rst->number[3] = num4;
        rst->number[4] = num5;
        rst->number[5] = num6;
        return rst;
    }
    __device__  BigInt* BigIntOper::CreatInt512(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, __uint64_t num5, __uint64_t num6, __uint64_t num7, __uint64_t num8, bool is_signed) { //512位
        return CreatInt512(num1, num2, num3, num4, num5, num6, num7, num8, is_signed, Mode::Private);
    }
    __device__  BigInt* BigIntOper::CreatInt512(__uint64_t num1, __uint64_t num2, __uint64_t num3, __uint64_t num4, __uint64_t num5, __uint64_t num6, __uint64_t num7, __uint64_t num8, bool is_signed, Mode mod)  { //512位
        int size = 8;
        BigInt* rst = VmCache::getBigInt();
        rst->mod = mod;
        rst->is_signed = is_signed;
        rst->size = size;
        rst->number[0] = num1;
        rst->number[1] = num2;
        rst->number[2] = num3;
        rst->number[3] = num4;
        rst->number[4] = num5;
        rst->number[5] = num6;
        rst->number[6] = num7;
        rst->number[7] = num8;
        return rst;
    }
*/
    __device__  BigInt* BigIntOper::add_w(const BigInt* a, const BigInt* b) {
        int n = max(a->size, b->size);
        BigInt* result = zero(n, a->is_signed || b->is_signed);

        U64 carry = 0;
        for (int i = 0; i < n || carry; ++i) {
            U64 sum = carry;
            if (i < a->size) {
                sum += a->number[i];
            }
            if (i < b->size) {
                sum += b->number[i];
            }
            carry = (sum < a->number[i]) || (carry && sum == a->number[i]);
            result->number[i] = sum;
        }
        return result;
    }

    __device__  BigInt* BigIntOper::add_w_change(BigInt* a, const BigInt* b) {
        int n = max(a->size, b->size);
        U64 carry = 0;
        for (int i = 0; i < n || carry; ++i) {
            U64 sum = carry;
            if (i < a->size) {
                sum += a->number[i];
            }
            if (i < b->size) {
                sum += b->number[i];
            }
            carry = (sum < a->number[i]) || (carry && sum == a->number[i]);
            a->number[i] = sum;
        }
        return a;
    }

    __device__  BigInt* BigIntOper::add(const BigInt* a, const BigInt* b) {
        return add_w(a, b);
    }

    __device__  BigInt* BigIntOper::mul_w(const BigInt* a, const BigInt* b) {
        int max_size = max(a->size, b->size);
        BigInt* result = zero(max_size, a->is_signed || b->is_signed);
        for (int i = 0; i < a->size; ++i) {
            __uint64_t carry = 0;
            for (int j = 0; j < b->size && (i + j) < max_size; ++j) {
                __uint128_t product = static_cast<__uint128_t>(a->number[i]) * b->number[j] + result->number[i + j] + carry;
                result->number[i + j] = static_cast<__uint64_t>(product);
                carry = static_cast<__uint64_t>(product >> 64);//取高位部分
            }
            if (i + b->size < result->size) {
                result->number[i + b->size] = carry;
            }
        }
        return result;
    }

    __device__  BigInt* BigIntOper::mul(const BigInt* a, const BigInt* b) {
        return mul_w(a, b);
    }

    __device__  BigInt* BigIntOper::sub_w(const BigInt* a, const BigInt* b) {
        int max_size = max(a->size, b->size);
        BigInt* result = zero(max_size, a->is_signed || b->is_signed);
        U64 carry = 0;

        // 假设 a 和 b 的大小相等
        for (int i = 0; i < a->size; i++) {
            U64 sum = a->number[i] + b->number[i] + carry;

            // 检查是否发生了溢出
            carry = (sum < a->number[i]) ? 1 : 0;

            // 将结果存入 result
            result->number[i] = sum;
        }
        return result;
    }

    __device__  BigInt* BigIntOper::sub_w_change(BigInt* a, const BigInt* b) {
        int max_size = max(a->size, b->size);
        U64 carry = 0;

        // 假设 a 和 b 的大小相等
        for (int i = 0; i < a->size; i++) {
            U64 sum = a->number[i] + b->number[i] + carry;

            // 检查是否发生了溢出
            carry = (sum < a->number[i]) ? 1 : 0;

            // 将结果存入 result
            a->number[i] = sum;
        }
        return a;
    }

    __device__  BigInt* BigIntOper::sub(const BigInt* a, const BigInt* b) {
        return sub_w(a, b);
    }

    __device__  bool BigIntOper::is_eq(const BigInt* a, const BigInt* b) {
        if (a->size != b->size) {
            return false;
        }
        for (int i=0; i < a->size; ++i) {
            if (a->number[i] != b->number[i]) {
                return false;
            }
        }
        return true;
    }

    __device__  bool BigIntOper::lt(const BigInt* a, const BigInt* b) {
        bool a_is_negative = isNegative(a);
        if (a_is_negative != isNegative(b)) {
            return a_is_negative;
        }
        if (a->size != b->size) {
            return (a->size < b->size) != a_is_negative;
        }
        for (int i = a->size - 1; i >= 0; --i) {
            if (a->number[i] != b->number[i]) {
                return (a->number[i] < b->number[i]) != a_is_negative;
            }
        }
        return false;
    }

    __device__  bool BigIntOper::is_neq(const BigInt* a, const BigInt* b) {
        return !is_eq(a, b);
    }

    __device__  bool BigIntOper::lte(const BigInt* a, const BigInt* b) {
        return lt(a, b) || is_eq(a, b);
    }

    __device__  bool BigIntOper::gt(const BigInt* a, const BigInt* b) {
        return !lte(a, b);
    }

    __device__  bool BigIntOper::gte(const BigInt* a, const BigInt* b) {
        return !lt(a, b);
    }

    __device__  BigInt* BigIntOper::xor_(const BigInt* a, const BigInt* b) {
        int max_size = max(a->size, b->size);
        BigInt* result = zero(max_size, a->is_signed);
        for (int i = 0; i < result->size; ++i) {
            __uint64_t ta = (i < a->size) ? a->number[i]: 0 ;
            __uint64_t tb = (i < b->size) ? b->number[i]: 0;
            result->number[i] = ta ^ tb;
        }
        return result;
    }

    __device__  BigInt* BigIntOper::and_(BigInt* a, BigInt* b) {
        int max_size = max(a->size, b->size);
        BigInt* result = zero(max_size, a->is_signed);
        for (int i = 0; i < result->size; ++i) {
            __uint64_t ta = (i < a->size) ? a->number[i]: 0 ;
            __uint64_t tb = (i < b->size) ? b->number[i]: 0;
            result->number[i] = ta & tb;
        }
        return result;
    }

    __device__  BigInt* BigIntOper::neg(const BigInt* a) {
        BigInt* result = zero(a->size, a->is_signed);
        for (int i = 0; i < result->size; ++i) {
            result->number[i] = ~a->number[i];
        }
        __uint64_t carry = 1;
        for (int i = 0; i < result->size; ++i) {
            __uint64_t sum = result->number[i] + carry;
            result->number[i] = sum;
            carry = (sum < result->number[i]) ? 1: 0; //检查是否有进位
            if (carry == 0) {
                break;
            }
        }
        return result;
    }

    __device__  BigInt* BigIntOper::shr_w(const BigInt* a, int shift) { //回环右移
        if (shift <= 0) {
            return clone(a);
        }
        shift = shift % (a->size*64);
        // 计算需要移位的块数和每个块内的位数
        int word_shift = shift / 64;  // 移动的 uint64_t 数量
        int bit_shift = shift % 64;   // 移动的位数

        // 临时数组存储移位结果
        BigInt* data = clone(a);
        int size = data->size;
        // 先进行整体块移动
        for (int i = 0; i < size - word_shift; ++i) {
            data->number[i] = data->number[i + word_shift];  // 移动块的内容
        }

        // 将剩余的高位块清零
        for (int i = size - word_shift; i < size; ++i) {
            data->number[i] = 0;  // 清零多余的高位块
        }

        // 再处理块内的位移
        if (bit_shift > 0) {
            for (int i = 0; i < size - 1; ++i) {
                data->number[i] = (data->number[i] >> bit_shift) | (data->number[i + 1] << (64 - bit_shift));  // 当前块右移并拼接下一个块的高位
            }
            // 处理最后一个块的右移
            data->number[size - 1] = data->number[size - 1] >> bit_shift;  // 只需右移最后一个块
        }

        return data;
    }

    __device__  BigInt* BigIntOper::shl_w(const BigInt* a, int shift) { //回环左移
        if (shift <= 0) {
            return clone(a);
        }
        shift = shift % (a->size*64);
        int word_shift = shift / 64;
        int bit_shift = shift % 64;
        BigInt* result = zero(a->size, a->is_signed, a->mod);
        // 按位移动
        for (int i = result->size - 1; i >= 0; --i) {
            if (i - word_shift >= 0) {
                result->number[i] = a->number[i - word_shift] << bit_shift;
                if (bit_shift > 0 && i - word_shift - 1 >= 0) {
                    // 将低位进位到高位
                    result->number[i] |= (a->number[i - word_shift - 1] >> (64 - bit_shift));
                }
            }
        }
        return result;
    }

    __device__  BigInt* BigIntOper::abs_w(const BigInt* num) {
        BigInt* a = clone(num);
        if (a->is_signed && isNegative(a)) {
            a = to_original(a);
        }
        return a;
    }

    __device__  BigInt* BigIntOper::abs(const BigInt* a) {
        return abs_w(a);
    }

    __device__  BigInt* BigIntOper::div_w(const BigInt* dividend, const BigInt* divisor) {
        if (!dividend->is_signed && !divisor->is_signed) { //无符号
            BigInt* quotient = zero(2, false, dividend->mod);  // 商
            BigInt* remainder = clone(quotient); // 余数
            BigInt* _dividend = clone(dividend);
            for (int i = 127; i >= 0; --i) {
                BigInt* tmp = remainder;
                remainder = shl_w(remainder, 1);
                destory(tmp);
                remainder->number[0] |= (dividend->number[1] >> 63);
                tmp = _dividend;
                _dividend = shl_w(_dividend, 1);
                destory(tmp);
                if (!compare_uint128(remainder, divisor)) {
                    remainder = subtract_uint128(remainder, divisor);
                    quotient->number[0] |= (1ULL << i);
                }
            }
            destory(remainder);
            destory(_dividend);
            return quotient;
        } else { //有符号
            bool negative_result = isNegative(dividend) != isNegative(divisor);
            // 计算被除数和除数的绝对值
            BigInt* abs_dividend = abs(dividend);
            BigInt* abs_divisor = abs(divisor);

            // 调用无符号除法进行计算
            abs_dividend->is_signed = false;
            abs_divisor->is_signed = false;
            BigInt* quotient = div_w(abs_dividend, abs_divisor);

            // 根据结果符号修正商
            if (negative_result) {
                quotient->number[0] = ~quotient->number[0] + 1;
                quotient->number[1] = ~quotient->number[1] + (quotient->number[0] == 0);
            }
            quotient->is_signed = dividend->is_signed;
            destory(abs_dividend);
            destory(abs_divisor);
            return quotient;
        }
    }

    __device__  BigInt* BigIntOper::div(const BigInt* a, const BigInt* b) {
        return div_w(a, b);
    }

    __device__  BigInt* BigIntOper::ternary(bool condition, BigInt* a, BigInt* b) {
        return condition?a:b;
    }

    __device__  void BigIntOper::destory(BigInt* a) { //必须要销毁
        if (nullptr == a) {
           return;
        }
        VmCache::returnBigInt(a);
    }

    // 打印大整数
    // 打印大整数
    __device__  void BigIntOper::print_bigint(const BigInt* num) {
        BigInt* a = clone(num);
        char buffer[512];
        int pos = 0;
        char temp[512] = {0};
        bool is_negative = false;

        // 处理有符号数
        if (a->is_signed && (a->number[a->size - 1] & (1ULL << 63))) {
            is_negative = true;
            // 处理负数，取反加1
            for (int i = 0; i < a->size; ++i) {
                a->number[i] = ~a->number[i];
            }
            U64 carry = 1;
            for (int i = 0; i < a->size; ++i) {
                a->number[i] += carry;
                if (a->number[i] != 0) carry = 0;
            }
        }

        // 使用基数10^9
        const U64 BASE = 1000000000ULL;
        while (a->size > 1 || a->number[0] != 0) {
            U64 remainder = divmod_bigint(a, BASE);
            // 将remainder转换为字符串，并存入temp数组
            for (int i = 0; i < 9; ++i) {
                temp[pos++] = '0' + remainder % 10;
                remainder /= 10;
            }
            // 防止死循环，确保大整数在每次迭代后缩小
            if (a->size == 0 && a->number[0] == 0) {
                break;
            }
        }
        // 反转并处理负号
        int start = 0;
        if (is_negative) buffer[start++] = '-';
        while (pos > 0 && temp[pos - 1] == '0') --pos;  // 去掉多余的前导0
        while (pos > 0) {
            buffer[start++] = temp[--pos];
        }
        if (start == 0) buffer[start++] = '0';  // 如果结果是0
        buffer[start] = '\0';  // 字符串结尾符
        printf("BigInt: %s\n", buffer);
    }

    __device__  void BigIntOper::print_hex(const BigInt* num) {
        int size = num->size;
        for (int i = size-1; i >= 0; i--) {
            printf("0x%016llx", (U64)num->number[i]);
            if (i!=0) {
                printf(",");
            }
        }
        printf("\n");
    }

    // 设备端模板函数，允许精度丢失的 BigInt 转普通整型（转化为64位以下的整形）
    template<typename To>
    __device__
    To BigIntOper::cast_lossy(const BigInt& bigInt) {
        return static_cast<To>(bigInt.number[0]);
    }

    __device__  Vec<Bool>* BigIntOper::to_bits_le(const BigInt* bigInt) {
        return to_bits_le(bigInt, Mode::Private);
    }

    __device__  Vec<Bool>* BigIntOper::to_bits_le(const BigInt* bigInt, Mode mode) {
        U32 len =  bigInt->size* 64;
        Vec<Bool>* vec = VmCache::getBoolStructLst(len);
        for (int i = 0; i < bigInt->size; i++) {
            U64 tmp = bigInt->number[i];
            for (int j = 0; j < 64; j++) {
                vec->push(Bool{bool(tmp & 0x1), mode});
                tmp >>= 1;
            }
        }
        return vec;
    }
    __device__  Vec<Bool>* BigIntOper::to_bits_be(const BigInt* bigInt, Mode mode) {
        Vec<Bool>* vec = to_bits_le(bigInt, mode);
        return vec->reverse();
    }
    __device__  Vec<Bool>* BigIntOper::to_bits_be(const BigInt* bigInt) {
        return to_bits_be(bigInt, Mode::Private);
    }

    __device__  Vec<skm::BigInt*>* BigIntOper::clone_vec(Vec<skm::BigInt*>* src) {
        Vec<skm::BigInt*>* newVec = VmCache::getBigIntLst(src->getSize());
        for (int i = 0; i < src->getSize(); i++) {
            newVec->set(i, clone(src->get(i)));
        }
        newVec->setSize(src->getSize());
        return newVec;
    }

    __device__  void BigIntOper::copy_from_slice(Vec<skm::BigInt*>* target,  Vec<skm::BigInt*> *src, u64 start, u64 len) {
        for (int i=0; i < len; i++) {
            target->set(i, src->get(i+start));
        }
    }
    __device__  void BigIntOper::copy_from_slice(Vec<skm::BigInt*>* target,  Vec<skm::BigInt*> *src) {
        copy_from_slice(target, src, 0, src->getSize());
    }

    __device__  Vec<skm::BigInt*>* BigIntOper::clone_from_slice( Vec<skm::BigInt*>* target , Vec<skm::BigInt*>* src) {
        return clone_from_slice(target, src, 0, src->getSize());
    }

    __device__  Vec<skm::BigInt*>* BigIntOper::clone_from_slice(Vec<skm::BigInt*>* target, Vec<skm::BigInt*>* src , u64 start, u64 len) {
        for (int i = 0; i < len; i++) {
            copy_to(src->get(i+start), target->get(i));
        }
        return target;
    }
}

