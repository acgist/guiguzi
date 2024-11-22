#include "./guiguzi/Guiguzi.hpp"

#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>

extern "C" {

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"

}

[[maybe_unused]] static void testPCM() {
    std::ifstream input;
    input.open("D:/tmp/audio.pcm", std::ios_base::binary);
    std::ofstream output;
    output.open("D:/tmp/audio.rnnoise.pcm", std::ios::trunc | std::ios_base::binary);
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
    std::vector<short> data;
    data.resize(480);
    // std::vector<char> data;
    // data.resize(960);
    guiguzi::Rnnoise rnnoise;
    if(!rnnoise.init()) {
        std::cout << "加载rnnoise失败\n";
        input.close();
        output.close();
        rnnoise.release();
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
    guiguzi::Rnnoise rnnoise(48000, 2, "mp3");
    // guiguzi::Rnnoise rnnoise(48000, 2, "opus");
    if(!rnnoise.init()) {
        std::cout << "加载rnnoise失败\n";
        rnnoise.release();
        return;
    }
    AVPacket       * packet    = av_packet_alloc();
    AVFormatContext* inputCtx  = avformat_alloc_context();
    AVFormatContext* outputCtx = avformat_alloc_context();
    #if _WIN32
    const char* input_file  = "D:/tmp/audio.mp3";
    const char* output_file = "D:/tmp/audio.rnnoise.mp3";
    #else
    const char* input_file  = "/data/guiguzi/audio.mp3";
    const char* output_file = "/data/guiguzi/audio.rnnoise.mp3";
    #endif
    if(avformat_open_input(&inputCtx, input_file, NULL, NULL) != 0) {
        std::cout << "打开音频输入文件失败\n";
        rnnoise.release();
        av_packet_free(&packet);
        avformat_close_input(&inputCtx);
        avformat_close_input(&outputCtx);
        return;
    }
    if(avformat_alloc_output_context2(&outputCtx, NULL, NULL, input_file) != 0) {
        std::cout << "打开音频输出文件失败\n";
        rnnoise.release();
        av_packet_free(&packet);
        avformat_close_input(&inputCtx);
        avformat_close_input(&outputCtx);
        return;
    }
    AVStream* stream = avformat_new_stream(outputCtx, avcodec_find_encoder(AV_CODEC_ID_MP3));
    // AVStream* stream = avformat_new_stream(outputCtx, avcodec_find_encoder(AV_CODEC_ID_OPUS));
    if(!stream) {
        std::cout << "打开音频流失败\n";
        rnnoise.release();
        av_packet_free(&packet);
        avformat_close_input(&inputCtx);
        avformat_close_input(&outputCtx);
        return;
    }
    if(avcodec_parameters_from_context(stream->codecpar, rnnoise.encodeCodecCtx) != 0) {
        std::cout << "拷贝音频配置失败\n";
        rnnoise.release();
        av_packet_free(&packet);
        avformat_close_input(&inputCtx);
        avformat_close_input(&outputCtx);
        return;
    }
    if(avio_open(&outputCtx->pb, output_file, AVIO_FLAG_WRITE) != 0) {
        std::cout << "打开音频输出文件失败\n";
        rnnoise.release();
        av_packet_free(&packet);
        avformat_close_input(&inputCtx);
        avformat_close_input(&outputCtx);
        return;
    }
    avformat_write_header(outputCtx, NULL);
    std::vector<char> out;
    std::vector<int64_t> dts;
    std::vector<int64_t> pts;
    while(av_read_frame(inputCtx, packet) == 0) {
        dts.push_back(packet->dts);
        pts.push_back(packet->pts);
        rnnoise.superSweet(packet->data, packet->size, out);
        av_packet_unref(packet);
        if(!out.empty()) {
            packet->dts = dts[0];
            packet->pts = pts[0];
            packet->data = reinterpret_cast<uint8_t*>(out.data());
            packet->size = out.size();
            packet->stream_index = stream->index;
            dts.erase(dts.begin());
            pts.erase(pts.begin());
            av_write_frame(outputCtx, packet);
            av_packet_unref(packet);
            out.clear();
        }
    }
    av_packet_unref(packet);
    av_write_trailer(outputCtx);
    rnnoise.release();
    av_packet_free(&packet);
    // avio_close(inputCtx->pb);
    // avformat_free_context(inputCtx);
    avformat_close_input(&inputCtx);
    // avio_close(outputCtx->pb);
    // avformat_free_context(outputCtx);
    avformat_close_input(&outputCtx);
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
