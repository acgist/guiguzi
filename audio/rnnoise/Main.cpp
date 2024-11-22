#include "./guiguzi/Guiguzi.hpp"

#include <chrono>
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
        std::cout << "打开音频输入文件失败\n";
        return;
    }
    if(!output.is_open()) {
        std::cout << "打开音频输出文件失败\n";
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
    AVPacket       * packet    = av_packet_alloc();
    AVFormatContext* formatCtx = avformat_alloc_context();
    if(avformat_open_input(&formatCtx, "D:/tmp/audio.mp3", NULL, NULL) < 0) {
    // if(avformat_open_input(&formatCtx, "D:/tmp/audio.mono.mp3", NULL, NULL) < 0) {
        std::cout << "打开音频输入文件失败\n";
        return;
    }
    std::ofstream output;
    output.open("D:/tmp/audio.rnnoise.mp3", std::ios::trunc | std::ios_base::binary);
    if(!output.is_open()) {
        std::cout << "打开音频输出文件失败\n";
        return;
    }
    guiguzi::Rnnoise rnnoise(48000, 2, "mp3");
    // guiguzi::Rnnoise rnnoise(48000, 1, "mp3");
    if(!rnnoise.init()) {
        std::cout << "加载rnnoise失败\n";
        return;
    }
    std::vector<char> out;
    while(av_read_frame(formatCtx, packet) == 0) {
        rnnoise.superSweet(packet->data, packet->size, out);
        if(!out.empty()) {
            output.write(out.data(), out.size());
            out.clear();
        }
        av_packet_unref(packet);
    }
    av_packet_unref(packet);
    output.close();
    rnnoise.release();
    av_packet_free(&packet);
    avformat_close_input(&formatCtx);
    avformat_free_context(formatCtx);
}

int main() {
    // testPCM();
    int i = 0;
    auto a = std::chrono::system_clock::now();
    // while(++i < 1000) { // 测试内存泄漏
        testFFmpeg();
    // }
    auto z = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count() << '\n';
    return 0;
}
