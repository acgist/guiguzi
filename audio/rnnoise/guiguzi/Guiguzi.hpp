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
 * aac
 * mp3
 * opus
 * 
 * 数据处理流程：
 * 解码->重采样->降噪->重采样->编码
 * 
 * 注意：第二次重采样直接使用复制数据算法，所以不支持fltp格式的aac，如果需要支持aac需要实现二次重采样。
 * 
 * @author acgist
 */
#ifndef GUIGUZI_RNNOISE_HPP
#define GUIGUZI_RNNOISE_HPP

#include <string>
#include <vector>
#include <cstdint>

struct DenoiseState;

struct AVFrame;
struct AVCodec;
struct AVPacket;
struct SwrContext;
struct AVCodecContext;
struct AVChannelLayout;

namespace guiguzi {

class Rnnoise {

public:
    bool inited{ false }; // 是否加载
    DenoiseState* denoise{ nullptr }; // 降噪对象
    float       * buffer_denoise{ nullptr }; // 降噪缓存
    std::vector<short> buffer_swr; // 重采样缓存（降噪使用）: 480次采样才降噪一次
    std::string format; // 输入输出格式
    size_t ar;          // 输出采样率
    size_t ac;          // 输出通道数
    int swr_ac;         // 重采样通道数
    int encoder_nb_samples{ 960 }; // 编码帧数
    int rnnoise_nb_samples{ 0   }; // 降噪帧数偏移
    AVFrame        * frame  { nullptr }; // 数据帧
    AVPacket       * packet { nullptr }; // 数据包
    const AVCodec  * decoder{ nullptr }; // 解码器
    const AVCodec  * encoder{ nullptr }; // 编码器
    SwrContext     * swrCtx { nullptr }; // 重采样上下文
    AVCodecContext * decodeCodecCtx{ nullptr }; // 解码器上下文
    AVCodecContext * encodeCodecCtx{ nullptr }; // 编码器上下文

public:
    bool init(); // 加载资源
    void sweet(char   * input); // 降噪
    void sweet(short  * input); // 降噪
    void sweet(float  * input); // 降噪
    void sweet(uint8_t* input); // 降噪
    bool superSweet(uint8_t* input, const size_t& size, std::vector<char>& out); // 封装格式降噪
    void release(); // 释放资源

public:
    Rnnoise(size_t ar = 48000, size_t ac = 1, std::string format = "pcm");
    virtual ~Rnnoise();

};

} // END OF guiguzi

#endif // END OF GUIGUZI_RNNOISE_HPP
