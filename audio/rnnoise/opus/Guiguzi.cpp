#include "Guiguzi.hpp"

#include <cstring>
#include <iostream>

#include "rnnoise.h"

extern "C" {

#include "opus/opus.h"

}

static const int RNNOISE_FRAME = 480; // 降噪处理采样数量

guiguzi::Rnnoise::Rnnoise(size_t ac, size_t ar, size_t hz, size_t bits) : ac(ac), ar(ar), hz(hz), bits(bits) {
}

guiguzi::Rnnoise::~Rnnoise() {
    this->release();
}

bool guiguzi::Rnnoise::init() {
    this->denoise = rnnoise_create(NULL);
    this->buffer_denoise = new float[RNNOISE_FRAME];
    this->per_sample = this->ar / 1000 * this->hz;
    this->per_size   = this->per_sample * this->bits / 8 * this->ac;
    this->buffer_rnnoise.resize(this->per_size * 4);
    int error_code = 0;
    this->decoder = opus_decoder_create(this->ar, this->ac, &error_code);
    if(!this->decoder) {
        std::cout << "打开解码器失败：" << error_code << '\n';
        return false;
    }
    opus_decoder_ctl(this->decoder, OPUS_SET_DTX(1));
    // opus_decoder_ctl(this->decoder, OPUS_SET_VBR(1));
    opus_decoder_ctl(this->decoder, OPUS_SET_INBAND_FEC(1));
    opus_decoder_ctl(this->decoder, OPUS_SET_LSB_DEPTH(this->bits));
    this->encoder = opus_encoder_create(this->ar, this->ac, OPUS_APPLICATION_VOIP, &error_code);
    // this->encoder = opus_encoder_create(this->ar, this->ac, OPUS_APPLICATION_AUDIO, &error_code);
    // this->encoder = opus_encoder_create(this->ar, this->ac, OPUS_APPLICATION_RESTRICTED_LOWDELAY, &error_code);
    if(!this->encoder) {
        std::cout << "打开编码器失败：" << error_code << '\n';
        return false;
    }
    opus_encoder_ctl(this->encoder, OPUS_SET_DTX(0));
    // opus_encoder_ctl(this->encoder, OPUS_SET_VBR(1));
    opus_encoder_ctl(this->encoder, OPUS_SET_BITRATE(64'000));
    opus_encoder_ctl(this->encoder, OPUS_SET_SIGNAL(OPUS_SIGNAL_VOICE));
    opus_encoder_ctl(this->encoder, OPUS_SET_INBAND_FEC(0));
    opus_encoder_ctl(this->encoder, OPUS_SET_LSB_DEPTH(this->bits));
    return true;
}

void guiguzi::Rnnoise::release() {
    if(this->denoise) {
        rnnoise_destroy(this->denoise);
        this->denoise = nullptr;
    }
    if(this->buffer_denoise) {
        delete[] this->buffer_denoise;
        this->buffer_denoise = nullptr;
    }
    if(this->decoder) {
        opus_decoder_destroy(this->decoder);
        this->decoder = nullptr;
    }
    if(this->encoder) {
        opus_encoder_destroy(this->encoder);
        this->encoder = nullptr;
    }
}

void guiguzi::Rnnoise::sweet(char* input) {
    this->sweet(reinterpret_cast<short*>(input));
}

void guiguzi::Rnnoise::sweet(short* input) {
    std::copy(input, input + RNNOISE_FRAME, this->buffer_denoise);
    this->sweet(this->buffer_denoise);
    std::copy(this->buffer_denoise, this->buffer_denoise + RNNOISE_FRAME, input);
}

void guiguzi::Rnnoise::sweet(uint8_t* input) {
    this->sweet(reinterpret_cast<short*>(input));
}

void guiguzi::Rnnoise::sweet(float* input) {
    rnnoise_process_frame(this->denoise, input, input);
}

bool guiguzi::Rnnoise::putSweet(uint8_t* input, const size_t& size) {
    if(size <= 0) {
        return false;
    }
    const int nb_samples = opus_decode(this->decoder, input, size, this->buffer_rnnoise.data() + this->rnnoise_end, this->per_sample, 0);
    if(nb_samples <= 0) {
        return false;
    }
    this->rnnoise_end += (nb_samples * this->ac);
    if(this->rnnoise_end >= this->buffer_rnnoise.size()) {
        // 数据缓存满了直接清空
        this->rnnoise_end = 0;
        this->rnnoise_pos = 0;
        return false;
    }
    while(this->rnnoise_end >= this->rnnoise_pos + RNNOISE_FRAME * this->ac) {
        // 重采样格式
        // packed: LRLRLRLR
        // planar: LLLLRRRR
        // 向下重采样
        for(size_t index = 0; index < RNNOISE_FRAME; ++index) {
            this->buffer_denoise[index] = this->buffer_rnnoise[this->rnnoise_pos + this->ac * index];
        }
        // 降噪
        this->sweet(this->buffer_denoise);
        // 向上重采样
        for(size_t index = 0; index < RNNOISE_FRAME; ++index) {
            if(this->ac == 1) {
                this->buffer_rnnoise[this->rnnoise_pos + this->ac * index] = this->buffer_denoise[index];
            } else {
                this->buffer_rnnoise[this->rnnoise_pos + this->ac * index    ] = this->buffer_denoise[index];
                this->buffer_rnnoise[this->rnnoise_pos + this->ac * index + 1] = this->buffer_denoise[index];
            }
        }
        this->rnnoise_pos += RNNOISE_FRAME * this->ac;
    }
    return true;
}

bool guiguzi::Rnnoise::getSweet(std::vector<char>& out, size_t& size) {
    if(this->rnnoise_end <= 0 || this->rnnoise_pos <= 0) {
        return false;
    }
    #ifdef __PCM__
    size = this->rnnoise_pos * sizeof(short);
    std::memcpy(out.data(), this->buffer_rnnoise.data(), size);
    this->rnnoise_end -= this->rnnoise_pos;
    this->rnnoise_pos -= this->rnnoise_pos;
    #else
    size = opus_encode(this->encoder, this->buffer_rnnoise.data(), this->per_sample, reinterpret_cast<unsigned char*>(out.data()), this->per_size);
    const int sample_size = this->per_sample * this->ac;
    this->rnnoise_end -= sample_size;
    this->rnnoise_pos -= sample_size;
    if(this->rnnoise_end > 0) {
        // 正常分包不会出现
        std::copy(this->buffer_rnnoise.begin() + sample_size, this->buffer_rnnoise.begin() + sample_size + this->rnnoise_end, this->buffer_rnnoise.begin());
    }
    #endif
    return true;
}
