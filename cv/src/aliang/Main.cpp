#include "Aliang.hpp"

#include "opencv2/opencv.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

static void initLogger() {
    std::vector<spdlog::sink_ptr> sinks{};
    // 开发日志
    #if defined(_DEBUG) || !defined(NDEBUG)
    sinks.reserve(2);
    auto stdoutColorSinkSPtr = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(stdoutColorSinkSPtr);
    #endif
    // 文件日志
    // auto dailyFileSinkSPtr = std::make_shared<spdlog::sinks::daily_file_sink_mt>("../logs/guiguzi.log", 0, 0, false, 7);
    // sinks.push_back(dailyFileSinkSPtr);
    // 默认日志
    auto logger = std::make_shared<spdlog::logger>("guiguziLogger", sinks.begin(), sinks.end());
    #if defined(_DEBUG) || !defined(NDEBUG)
    logger->set_level(spdlog::level::debug);
    #else
    logger->set_level(spdlog::level::info);
    #endif
    logger->flush_on(spdlog::level::warn);
    logger->set_pattern("[%D %T] [%L] [%t] [%s:%#] %v");
    spdlog::set_default_logger(logger);
    SPDLOG_DEBUG(R"(
        
        天上剑仙三百万，见我也须尽低眉。
    )");
}

int main() {
    #if _WIN32
    system("chcp 65001");
    #endif
    initLogger();
    std::vector<std::string> classes{ "helment", "person" };
    guiguzi::Detection detection("D:/tmp/helmet/best.onnx", "aliang-detection", classes);
    // auto input = cv::imread("D:/tmp/helmet/train/1.jpg");
    // auto input = cv::imread("D:/tmp/helmet/val/34.jpg");
    // auto input = cv::imread("D:/tmp/helmet/1.jpg");
    // detection.detection(input);
    // cv::imshow("image", input);
    // cv::waitKey(0);
    guiguzi::Recognition recognition(
        "D:/tmp/face/yoloface_8n.onnx", "aliang-face",
        "D:/tmp/face/arcface_w600k_r50.onnx", "aliang-face-feature"
    );
    recognition.storage("yusheng", {
        "D:/tmp/face/y1.jpg",
        "D:/tmp/face/y6.jpg"
    });
    recognition.storage("zhong", {
        "D:/tmp/face/z.jpg"
    });
    cv::VideoCapture capture(0);
    if(!capture.isOpened()) {
        return -1;
    }
    cv::Mat frame;
    while(true) {
        capture >> frame;
        detection.detection(frame, recognition);
        cv::imshow("image", frame);
        auto key = cv::waitKey(1);
        if(key == 'q') {
            break;
        }
    }
    SPDLOG_DEBUG("完成");
    return 0;
}