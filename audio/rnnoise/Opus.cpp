#include "opus/Guiguzi.hpp"

#include <chrono>
#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>

static void testEncode() {
    std::ifstream input;
    input.open("D:/tmp/audio.pcm", std::ios_base::binary);
    std::ofstream output;
    output.open("D:/tmp/audio.rnnoise.opus", std::ios::trunc | std::ios_base::binary);
    std::vector<char> out;
    std::vector<char> data;
    out.resize(16 * 1024);
    size_t size = 0;
    data.resize(1920); // 48000 / 1000 * 10 * 2 * 2
    guiguzi::Rnnoise rnnoise;
    rnnoise.init();
    while(input.read(data.data(), data.size())) {
        rnnoise.rnnoise_pos = 960;
        rnnoise.buffer_rnnoise.resize(960);
        std::memcpy(rnnoise.buffer_rnnoise.data(), data.data(), 1920);
        if(rnnoise.getSweet(out, size)) {
            std::cout << size << '\n';
            int buffer_size = size;
            output.write(reinterpret_cast<char*>(&buffer_size), sizeof(buffer_size));
            output.write(out.data(), size);
        }
    }
    input.close();
    output.close();
    rnnoise.release();
}

static void testOpus() {
    std::ifstream input;
    // input.open("D:/tmp/audio.opus", std::ios_base::binary);
    input.open("D:/tmp/audio.pack.1.opus", std::ios_base::binary);
    // input.open("D:/tmp/audio.pack.2.opus", std::ios_base::binary);
    std::ofstream output;
    output.open("D:/tmp/audio.rnnoise.opus", std::ios::trunc | std::ios_base::binary);
    if(!input.is_open()) {
        std::cout << "打开音频输入文件失败\n";
        input.close();
        return;
    }
    if(!output.is_open()) {
        std::cout << "打开音频输出文件失败\n";
        output.close();
        return;
    }
    // guiguzi::Rnnoise rnnoise(2, 48000, 10, 16);
    guiguzi::Rnnoise rnnoise(1, 48000, 20, 16);
    if(!rnnoise.init()) {
        std::cout << "加载rnnoise失败\n";
        input.close();
        output.close();
        rnnoise.release();
        return;
    }
    std::vector<char> out;
    std::vector<char> data;
    out.resize(16 * 1024);
    size_t size = 0;
    int buffer_size = 0;
    while(input.read(reinterpret_cast<char*>(&buffer_size), sizeof(buffer_size))) {
        data.resize(buffer_size);
        input.read(data.data(), buffer_size);
        rnnoise.putSweet(reinterpret_cast<uint8_t*>(data.data()), buffer_size);
        if(rnnoise.getSweet(out, size)) {
            // std::cout << "====" << size << '\n';
            output.write(out.data(), size);
            size = 0;
        }
    }
    input.close();
    output.close();
    rnnoise.release();
}

int main() {
    auto a = std::chrono::system_clock::now();
    // for(int i = 0; i < 1000; ++i) {
        testOpus();
    // }
    // testEncode();
    auto z = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count() << '\n';
    return 0;
}