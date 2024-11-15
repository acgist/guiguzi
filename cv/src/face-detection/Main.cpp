#include "ggz/Logger.hpp"

#include "spdlog/spdlog.h"

#include "ggz/cv/FaceDetection.hpp"

[[maybe_unused]] static void testOpenCV() {
    ggz::opencv_face_detection("D:/tmp/face/yolov5su.onnx", "D:/tmp/F4.jpg");
    // ggz::opencv_face_detection("D:/gitee/ggz/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg");
    // ggz::opencv_face_detection("D:/gitee/ggz/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg");
}

[[maybe_unused]] static void testLibTorch() {
    try {
        ggz::libtorch_face_detection("D:/tmp/face/yolo11n.torchscript", "D:/tmp/face/xx.jpg");
        ggz::libtorch_face_detection("D:/tmp/face/yolo11n.torchscript", "D:/tmp/face/zidane.jpg");
        // ggz::libtorch_face_detection("D:/tmp/face/yolo11n.torchscript", "D:/tmp/F4.jpg");
        // ggz::libtorch_face_detection("D:/tmp/face/yolov5s.pt", "D:/tmp/F4.jpg");
        // ggz::libtorch_face_detection("D:/tmp/face/yolov5su.pt", "D:/tmp/F4.jpg");
    } catch(const std::exception& e) {
        SPDLOG_ERROR("testLibTorch : {}", e.what());
    }
}

int main() {
    ggz::logger::init();
    // testOpenCV();
    testLibTorch();
    ggz::logger::shutdown();
    return 0;
}
