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
    int error_code = 0;
    this->decoder = opus_decoder_create(this->ar, this->ac, &error_code);
    if(!this->decoder) {
        std::cout << "打开解码器失败：" << error_code << '\n';
        return false;
    }
    this->per_sample = this->ar / 1000 * this->hz;
    this->per_size   = this->per_sample * this->bits / 8 * this->ac;
    // opus_decoder_ctl(this->decoder, OPUS_SET_LSB_DEPTH(this->bits));
    this->encoder = opus_encoder_create(this->ar, this->ac, OPUS_APPLICATION_VOIP, &error_code);
    // this->encoder = opus_encoder_create(this->ar, this->ac, OPUS_APPLICATION_RESTRICTED_LOWDELAY, &error_code);
    if(!this->encoder) {
        std::cout << "打开编码器失败：" << error_code << '\n';
        return false;
    }
    // opus_encoder_ctl(this->encoder, OPUS_SET_VBR(1));
    // opus_encoder_ctl(this->encoder, OPUS_SET_DTX(0));
    // opus_encoder_ctl(this->encoder, OPUS_SET_SIGNAL(OPUS_SIGNAL_VOICE));
    // opus_encoder_ctl(this->encoder, OPUS_SET_BITRATE(128'000));
    // opus_encoder_ctl(this->encoder, OPUS_SET_LSB_DEPTH(this->bits));
    // opus_encoder_ctl(this->encoder, OPUS_SET_INBAND_FEC(0));
    // opus_encoder_ctl(this->encoder, OPUS_SET_COMPLEXITY(8));
    // opus_encoder_ctl(this->encoder, OPUS_SET_VBR_CONSTRAINT(1));
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
    const int remaining_size = this->buffer_rnnoise.size();
    this->buffer_rnnoise.resize(remaining_size + this->per_size);
    const int nb_samples = opus_decode(this->decoder, input, size, this->buffer_rnnoise.data(), this->per_sample, 0);
    if(nb_samples <= 0) {
        this->buffer_rnnoise.resize(remaining_size);
        return false;
    }
    // 删除多余数据
    this->buffer_rnnoise.resize(remaining_size + nb_samples * this->ac);
    while(this->buffer_rnnoise.size() >= this->rnnoise_pos + RNNOISE_FRAME * this->ac) {
        for(size_t index = 0; index < RNNOISE_FRAME * this->ac; index += this->ac) {
            this->buffer_denoise[index / this->ac] = this->buffer_rnnoise[index];
        }
        this->sweet(this->buffer_denoise);
        for(size_t index = 0; index < RNNOISE_FRAME * this->ac; index += this->ac) {
            this->buffer_rnnoise[index] = this->buffer_denoise[index / this->ac];
            if(this->ac == 2) {
                this->buffer_rnnoise[index + 1] = this->buffer_denoise[index / this->ac];
            }
        }
        this->rnnoise_pos += RNNOISE_FRAME * this->ac;
    }
    return true;
}

bool guiguzi::Rnnoise::getSweet(std::vector<char>& out) {
    if(this->rnnoise_pos <= 0) {
        return false;
    }
    #ifdef __PCM__
    out.resize(this->per_size);
    std::memcpy(out.data(), this->buffer_rnnoise.data(), this->per_size);
    this->buffer_rnnoise.clear();
    this->rnnoise_pos = 0;
    #else
    out.resize(this->per_size);
    const int size = opus_encode(this->encoder, this->buffer_rnnoise.data(), this->per_sample, reinterpret_cast<unsigned char*>(out.data()), this->per_size);
    out.resize(size);
    if(size > 0) {
        this->buffer_rnnoise.erase(this->buffer_rnnoise.begin(), this->buffer_rnnoise.begin() + (this->per_sample * this->ac));
        this->rnnoise_pos -= this->per_sample * this->ac;
    }
    #endif
    return true;
}
