/**
 * 降噪音频格式：
 * 格式：PCM S16
 * 位深：16bit
 * 通道数：1
 * 采样率：48000
 * 
 * 数据大小计算：
 * ar    * ac * bits / 8 / 1000 * 10
 * 48000 * 1  * 16   / 8 / 1000 * 10 = 960 byte = 480 short
 * 
 * 封装格式支持：
 * opus
 * 
 * 数据处理流程：
 * 解码->重采样->降噪->重采样->编码
 * 
 * @author acgist
 */
#ifndef GUIGUZI_RNNOISE_HPP
#define GUIGUZI_RNNOISE_HPP

#include <string>
#include <vector>
#include <cstdint>

struct DenoiseState;

struct OpusDecoder;
struct OpusEncoder;

namespace guiguzi {

class Rnnoise {

public:
    DenoiseState*      denoise       { nullptr }; // 降噪对象
    float       *      buffer_denoise{ nullptr }; // 降噪缓存
    std::vector<short> buffer_rnnoise;            // 降噪缓存：默认16bit
    size_t ac  { 0 }; // 通道数
    size_t ar  { 0 }; // 采样率
    size_t hz  { 0 }; // 采样频率（毫秒）
    size_t bits{ 0 }; // 位深
    size_t per_size   { 0 }; // 采样大小：采样数量 * 位深 / 8 * 通道数
    size_t per_sample { 0 }; // 采样数量：采样率 / 1000 * 采样频率
    size_t end_sample { 0 }; // 尾部偏移
    size_t rnnoise_pos{ 0 }; // 降噪偏移
    OpusDecoder* decoder{ nullptr }; // 解码器
    OpusEncoder* encoder{ nullptr }; // 解码器

public:
    bool init();    // 加载资源
    void release(); // 释放资源
    void sweet(char   * input); // 降噪
    void sweet(short  * input); // 降噪
    void sweet(float  * input); // 降噪
    void sweet(uint8_t* input); // 降噪
    bool putSweet(uint8_t* input, const size_t& size);   // 封装格式降噪
    bool getSweet(std::vector<char>& out, size_t& size); // 读取封装数据

public:
    Rnnoise(size_t ac = 2, size_t ar = 48000, size_t hz = 10, size_t bits = 16);
    virtual ~Rnnoise();

};

} // END OF guiguzi

#endif // END OF GUIGUZI_RNNOISE_HPP
