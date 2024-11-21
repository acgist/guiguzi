#include "Guiguzi.hpp"

#include <vector>
#include <fstream>
#include <iostream>

#include "rnnoise.h"

// #define FF_API_OLD_CHANNEL_LAYOUT true

extern "C" {

#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswresample/swresample.h"

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
    } else if("mp3" == format) {
        codecId = AV_CODEC_ID_MP3;
    } else if("pcma" == format) {
        // PCMA (G.711 a-law)
        codecId = AV_CODEC_ID_PCM_ALAW;
    } else if("pcmu" == format) {
        // PCMU (G.711 μ-law)
        codecId = AV_CODEC_ID_PCM_MULAW;
    } else if("opus" == format) {
        codecId = AV_CODEC_ID_OPUS;
    } else {
        std::cout << "不支持的编码格式：" << format << '\n';
        return false;
    }
    // 数据帧
    this->frame = av_frame_alloc();
    // 数据包
    this->packet = av_packet_alloc();
    // 解码器
    this->decoder = avcodec_find_decoder(codecId); // 查找解码器
    this->decodeCodecCtx = avcodec_alloc_context3(this->decoder); // 创建解码器上下文
    if(!this->decodeCodecCtx) {
        std::cout << "创建解码器上下文失败：" << format << '\n';
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
    // 重采样 48000 PCM 单声道
    this->swrCtx = swr_alloc();
    if(!this->swrCtx) {
        std::cout << "创建重采样失败\n";
        return false;
    }
    auto channel_layout = this->ac == 1 ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;
    av_opt_set_channel_layout(this->swrCtx, "in_channel_layout",  channel_layout,                    0);
    av_opt_set_channel_layout(this->swrCtx, "out_channel_layout", AV_CH_LAYOUT_MONO,                 0);
    av_opt_set_int           (this->swrCtx, "in_sample_rate",     this->ar,                          0);
    av_opt_set_int           (this->swrCtx, "out_sample_rate",    48000,                             0);
    av_opt_set_sample_fmt    (this->swrCtx, "in_sample_fmt",      this->decodeCodecCtx->sample_fmt,  0);
    av_opt_set_sample_fmt    (this->swrCtx, "out_sample_fmt",     AV_SAMPLE_FMT_S16,                 0);
    this->swr_channels = av_get_channel_layout_nb_channels(AV_CH_LAYOUT_MONO);
    if(swr_init(this->swrCtx) != 0) {
        std::cout << "初始化重采样失败\n";
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
    if(this->swrCtx) {
        swr_close(this->swrCtx);
        swr_free(&this->swrCtx);
        this->swrCtx = nullptr;
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

bool guiguzi::Rnnoise::superSweet(uint8_t* input, size_t& length, std::vector<char>& outda) {
    this->packet->data = input;
    this->packet->size = length;
    if(avcodec_send_packet(this->decodeCodecCtx, this->packet) != 0) {
        std::cout << "音频解码请求失败\n";
        return false;
    }
    // av_packet_unref(this->packet); // 不用解除
    int pos = 0; // 当前降噪偏移 = PCM累计采样数量 = 次数 * 960 byte = 次数 * 480 short
    const int remaining_size = this->buffer_swr.size(); // 记录旧的没有降噪剩余数据
    while(avcodec_receive_frame(this->decodeCodecCtx, this->frame) == 0) {
        const int out_buffer_size = av_samples_get_buffer_size(NULL, this->swr_channels, this->frame->nb_samples, AV_SAMPLE_FMT_S16, 0);
        this->buffer_swr.resize(remaining_size + out_buffer_size);
        uint8_t* out = reinterpret_cast<uint8_t*>(this->buffer_swr.data() + remaining_size);
        // 重采样: 不用判断S16 S16P FLTP
        const int size = swr_convert(this->swrCtx, &out, this->frame->nb_samples, const_cast<const uint8_t**>(this->frame->data), this->frame->nb_samples);
        this->buffer_swr.resize(remaining_size + size);
        while(this->buffer_swr.size() > 480 + pos) {
            // 降噪
            this->sweet(this->buffer_swr.data() + pos);
            pos += 480;
        }
        av_frame_unref(this->frame);
    }
    av_frame_unref(this->frame);
    this->frame->format      = AV_SAMPLE_FMT_S16;
    this->frame->nb_samples  = pos;
    this->frame->sample_rate = 48000;
    if(this->ac == 1) {
        this->frame->ch_layout = AV_CHANNEL_LAYOUT_MONO;
    } else if(this->ac == 2) {
        this->frame->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
    } else {
        // 没有这种情况
    }
    // 重新分配大小
    av_frame_get_buffer(this->frame, 0);
    auto frame_data = reinterpret_cast<short*>(this->frame->data[0]);
    if(this->ac == 1) {
        std::memcpy(frame_data, this->buffer_swr.data(), pos * 2);
    } else {
        // 不用重采样直接复制：不存在两个通道数据不一致的情况
        for(int i = 0; i < pos; ++i) {
            frame_data[2 * i]     = this->buffer_swr.data()[i];
            frame_data[2 * i + 1] = this->buffer_swr.data()[i];
        }
    }
    if(avcodec_send_frame(this->encodeCodecCtx, this->frame) != 0) {
        av_frame_unref(this->frame);
        return false;
    }
    av_frame_unref(this->frame);
    while(avcodec_receive_packet(this->encodeCodecCtx, this->packet) == 0) {
        length = this->packet->size;
        outda.resize(this->packet->size);
        std::memcpy(outda.data(), this->packet->data, this->packet->size);
        av_packet_unref(this->packet);
    }
    av_packet_unref(this->packet);
    // 删除已经降噪数据
    this->buffer_swr.erase(this->buffer_swr.begin(), this->buffer_swr.begin() + pos);
    return true;
}
