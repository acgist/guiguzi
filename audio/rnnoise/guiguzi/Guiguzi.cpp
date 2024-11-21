#include "Guiguzi.hpp"

#include <iostream>

#include "rnnoise.h"

// #define FF_API_OLD_CHANNEL_LAYOUT true

extern "C" {

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"

}

static const int FRAME = 480;

guiguzi::Rnnoise::Rnnoise(size_t ar, size_t ac, size_t bits, std::string format) : ar(ar), ac(ac), bits(bits), format(format) {
}

guiguzi::Rnnoise::~Rnnoise() {
    this->release();
}

bool guiguzi::Rnnoise::init() {
    if(!this->denoise) {
        this->denoise = rnnoise_create(NULL);
    }
    if(!this->buffer_denoise) {
        this->buffer_denoise = new float[FRAME];
    }
    if(format == "pcm") {
        return true;
    }
    if(!this->buffer_codec) {
        // ar    * ac * bits / 8 / 1000 * 10
        // 48000 * 1  * 16   / 8 / 1000 * 10 = 960 byte = 480 short
        const int size = this->ar * this->ac * this->bits / 8 / 1000 * 10;
        this->buffer_codec = new float[size];
    }
    AVCodecID codecId;
    AVSampleFormat sampleFormat = AV_SAMPLE_FMT_S16;
    if("aac" == format) {
        codecId = AV_CODEC_ID_AAC;
        sampleFormat = AV_SAMPLE_FMT_FLTP;
    } else if("mp3") {
        codecId = AV_CODEC_ID_MP3;
    } else if("pcma") {
        // PCMA (G.711 a-law)
        codecId = AV_CODEC_ID_PCM_ALAW;
    } else if("pcmu") {
        // PCMU (G.711 μ-law)
        codecId = AV_CODEC_ID_PCM_MULAW;
    } else if("opus") {
        codecId = AV_CODEC_ID_OPUS;
    } else {
        std::cout << "不支持的编码格式：" << format << '\n';
        return false;
    }
    this->frame = av_frame_alloc();  // 解码数据帧
    // 48000 PCM 单声道
    this->frame->format     = AV_SAMPLE_FMT_S16P;
    this->frame->nb_samples = 48000;
    this->frame->ch_layout  = AV_CHANNEL_LAYOUT_MONO;
    this->packet = av_packet_alloc(); // 解码数据包
    // 解码器
    this->decoder = avcodec_find_decoder(codecId); // 查找解码器
    this->decodeCodecCtx = avcodec_alloc_context3(this->decoder); // 创建解码器上下文
    // avcodec_alloc_context3(NULL); // TODO: null 区别
    if(!this->decodeCodecCtx) {
        std::cout << "创建解码器上下文失败：" << format << '\n';
        return false;
    }
    this->decodeCodecCtx->sample_rate = this->ar;
    if(this->ac == 1) {
        this->decodeCodecCtx->ch_layout = AV_CHANNEL_LAYOUT_MONO;
    } else if(this->ac == 2) {
        this->decodeCodecCtx->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
    } else {
        std::cout << "不支持的通道数：" << this->ac << '\n';
        return false;
    }
    if(avcodec_open2(this->decodeCodecCtx, this->decoder, nullptr) < 0) {
        std::cout << "打开解码器失败：" << format << '\n';
        return false;
    }
    // 编码器
    this->encoder = avcodec_find_encoder(codecId); // 查找编码器
    this->encodeCodecCtx = avcodec_alloc_context3(this->encoder); // 创建编码器上下文
    if(!this->encodeCodecCtx) {
        std::cout << "创建编码器上下文失败：" << format << '\n';
        return false;
    }
    this->encodeCodecCtx->sample_fmt  = sampleFormat;
    this->encodeCodecCtx->sample_rate = this->ar;
    if(this->ac == 1) {
        this->encodeCodecCtx->ch_layout = AV_CHANNEL_LAYOUT_MONO;
    } else if(this->ac == 2) {
        this->encodeCodecCtx->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
    } else {
        std::cout << "不支持的通道数：" << this->ac << '\n';
        return false;
    }
    if(avcodec_open2(this->encodeCodecCtx, this->encoder, nullptr) < 0) {
        std::cout << "打开编码器失败：" << format << '\n';
        return false;
    }
    return true;
}

void guiguzi::Rnnoise::release() {
    if(this->denoise) {
        rnnoise_destroy(this->denoise);
        this->denoise = nullptr;
    }
    if(this->buffer_codec) {
        delete[] this->buffer_codec;
        this->buffer_codec = nullptr;
    }
    if(this->buffer_denoise) {
        delete[] this->buffer_denoise;
        this->buffer_denoise = nullptr;
    }
    if(this->frame) {
        av_frame_free(&this->frame);
        this->frame = nullptr;
    }
    if(this->packet) {
        av_packet_free(&this->packet);
        this->packet = nullptr;
    }
    if(this->decodeCodecCtx) {
        avcodec_close(this->decodeCodecCtx);
        avcodec_free_context(&this->decodeCodecCtx);
        this->decodeCodecCtx = nullptr;
    }
    if(this->encodeCodecCtx) {
        avcodec_close(this->encodeCodecCtx);
        avcodec_free_context(&this->encodeCodecCtx);
        this->encodeCodecCtx = nullptr;
    }
}

void guiguzi::Rnnoise::sweet(char* input) {
    this->sweet(reinterpret_cast<short*>(input));
}

void guiguzi::Rnnoise::sweet(short* input) {
    std::copy_n(input, FRAME, this->buffer_denoise);
    this->sweet(this->buffer_denoise);
    std::copy_n(this->buffer_denoise, FRAME, input);
}

void guiguzi::Rnnoise::sweet(uint8_t* input) {
    this->sweet(reinterpret_cast<short*>(input));
}

void guiguzi::Rnnoise::sweet(float* input) {
    rnnoise_process_frame(this->denoise, input, input);
}

bool guiguzi::Rnnoise::superSweet(uint8_t* input, size_t& length) {
    /*
     * 不用调用av_frame_unref和av_packet_unref
     */
    this->packet->data = input;
    this->packet->size = length;
    if(avcodec_send_packet(this->decodeCodecCtx, this->packet) < 0) {
        std::cout << "音频解码请求失败\n";
        return false;
    }
    // WAIT
    if(avcodec_receive_frame(this->decodeCodecCtx, this->frame) < 0) {
        std::cout << "音频解码读取失败" << '\n';
        return false;
    } else {
        auto data = this->frame->data;
        int out_buffer_size = av_samples_get_buffer_size(NULL, 2, this->frame->nb_samples, AV_SAMPLE_FMT_S16P, 1);
        uint32_t* x = reinterpret_cast<uint32_t*>(*data);
        for(int i = 0; i < out_buffer_size / 4; ++i) {
            x[i] = x[i] & 0x0000FFFF;
        }
        for(int i = 0; i < out_buffer_size; i += 960) {
            if(i + 960 > out_buffer_size) {
                break;
            }
            sweet((*data) + i);
        }
    }
    if(avcodec_send_frame(this->encodeCodecCtx, this->frame) < 0) {
        std::cout << "音频编码请求失败\n";
        return false;
    }
    // WAIT
    if(avcodec_receive_packet(this->encodeCodecCtx, this->packet) < 0) {
        std::cout << "音频编码读取失败" << '\n';
        return false;
    } else {
        input  = this->packet->data;
        // std::memcpy(input, this->packet->data, this->packet->size);
        length = this->packet->size;
    }
    return true;
}
