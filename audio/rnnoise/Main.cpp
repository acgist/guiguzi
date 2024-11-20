#include "./guiguzi/Guiguzi.hpp"

#include <vector>
#include <fstream>
#include <iostream>

extern "C" {

#include "libavformat/avformat.h"

}

[[maybe_unused]] static void testPCM() {
    std::ifstream input;
    input.open("D:/tmp/audio.pcm", std::ios_base::binary);
    std::ofstream output;
    output.open("D:/tmp/audio.rnnoise.pcm", std::ios::trunc | std::ios_base::binary);
    if(!input.is_open()) {
        return;
    }
    if(!output.is_open()) {
        return;
    }
    std::vector<short> data;
    data.resize(480);
    // std::vector<char> data;
    // data.resize(960);
    guiguzi::Rnnoise rnnoise;
    if(!rnnoise.init()) {
        std::cout << "加载rnnoise失败\n";
        return;
    }
    while(input.read(reinterpret_cast<char*>(data.data()), 960)) {
        rnnoise.sweet(data.data());
        output.write(reinterpret_cast<char*>(data.data()), 960);
    }
    input.close();
    output.close();
    rnnoise.release();
}

[[maybe_unused]] static void testFFmpeg() {
    const char* file   = "D:/tmp/audio.mp3";
    AVPacket       * packet    = av_packet_alloc();
    AVFormatContext* formatCtx = avformat_alloc_context();
    if(avformat_open_input(&formatCtx, file, NULL, NULL) < 0) {
        std::cout << "打开文件失败：" << file << '\n';
        return;
    }
    av_dump_format(formatCtx, 0, file, 0);
    avformat_find_stream_info(formatCtx, nullptr);
    std::ofstream output;
    output.open("D:/tmp/audio.rnnoise.mp3", std::ios::trunc | std::ios_base::binary);
    if(!output.is_open()) {
        return;
    }
    guiguzi::Rnnoise rnnoise(48000, 2, 16, "mp3");
    if(!rnnoise.init()) {
        std::cout << "加载rnnoise失败\n";
        return;
    }
    while(av_read_frame(formatCtx, packet) >= 0) {
        size_t size = packet->size;
        bool ret = rnnoise.superSweet(packet->data, size);
        output.write(reinterpret_cast<char*>(packet->data), size);
        av_packet_unref(packet);
    }
    output.close();
    rnnoise.release();
    av_packet_free(&packet);
    avformat_close_input(&formatCtx);
    avformat_free_context(formatCtx);
}

int main() {
    // testPCM();
    int i = 0;
    // while(++i < 1000) {
        testFFmpeg();
    // }
    return 0;
}
