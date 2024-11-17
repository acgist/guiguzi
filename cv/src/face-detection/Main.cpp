#include "guiguzi/Logger.hpp"

#include "spdlog/spdlog.h"

#include "guiguzi/cv/FaceDetection.hpp"

[[maybe_Unused]] static void testOnnx() {
    guiguzi::onnx_face_detection();
}

[[maybe_unused]] static void testOpenCV() {
    guiguzi::opencv_face_detection("D:/tmp/face/yolo11n.onnx", "D:/tmp/F4.jpg");
    // guiguzi::opencv_face_detection("D:/tmp/face/yolov5su.onnx", "D:/tmp/F4.jpg");
    // guiguzi::opencv_face_detection("D:/tmp/face/yolov5su.onnx", "D:/tmp/face/zidane.jpg");
    // guiguzi::opencv_face_detection("D:/gitee/guiguzi/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg");
    // guiguzi::opencv_face_detection("D:/gitee/guiguzi/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg");
}

[[maybe_unused]] static void testLibTorch() {
    try {
        guiguzi::libtorch_face_detection("D:/tmp/face/yolo11n.torchscript", "D:/tmp/F4.jpg");
        guiguzi::libtorch_face_detection("D:/tmp/face/yolo11n.torchscript", "D:/tmp/face/xx.jpg");
        guiguzi::libtorch_face_detection("D:/tmp/face/yolo11n.torchscript", "D:/tmp/face/zidane.jpg");
        // guiguzi::libtorch_face_detection("D:/tmp/face/yolov5s.pt", "D:/tmp/F4.jpg");
        // guiguzi::libtorch_face_detection("D:/tmp/face/yolov5su.pt", "D:/tmp/F4.jpg");
    } catch(const std::exception& e) {
        SPDLOG_ERROR("testLibTorch : {}", e.what());
    }
}

int main() {
    guiguzi::logger::init();
    testOnnx();
    // testOpenCV();
    // testLibTorch();
    guiguzi::logger::shutdown();
    return 0;
}
