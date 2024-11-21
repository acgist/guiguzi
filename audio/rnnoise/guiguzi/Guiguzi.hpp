/**
 * FFmpeg解码 -> Rnnoise降噪 -> FFmpeg编码
 * 
 * 降噪音频：
 * 格式：PCM
 * 位深：16bit
 * 通道数：1
 * 采样率：48000
 * 
 * 封装格式：
 * aac
 * mp3
 * pcm
 * pcma
 * pcmu
 * opus
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

namespace guiguzi {

class Rnnoise {

private:
    DenoiseState* denoise{ nullptr };
    float       * buffer_codec  { nullptr }; // 解码缓存
    float       * buffer_denoise{ nullptr }; // 降噪缓存
    std::vector<short> buffer_swr; // 重采样缓存
    size_t ar;          // 采样率
    size_t ac;          // 通道数
    size_t bits;        // 位深
    std::string format; // 格式
    int swr_channels;   // 重采样通道数
    AVFrame        * frame { nullptr }; // 数据帧
    AVPacket       * packet{ nullptr }; // 数据包
    SwrContext     * swrCtx{ nullptr }; // 重采样
    const AVCodec  * decoder       { nullptr }; // 编码器
    AVCodecContext * decodeCodecCtx{ nullptr }; // codec上下文
    const AVCodec  * encoder       { nullptr }; // 编码器
    AVCodecContext * encodeCodecCtx{ nullptr }; // codec上下文

public:
    bool init();                // 加载资源
    void sweet(char   * input); // 降噪
    void sweet(short  * input); // 降噪
    void sweet(float  * input); // 降噪
    void sweet(uint8_t* input); // 降噪
    bool superSweet(uint8_t* input, size_t& length, std::vector<char>& out); // 封装格式降噪
    void release();             // 释放资源

public:
    Rnnoise(size_t ar = 48000, size_t ac = 1, size_t bits = 16, std::string format = "opus");
    virtual ~Rnnoise();

};

} // END OF guiguzi

#endif // END OF GUIGUZI_RNNOISE_HPP
