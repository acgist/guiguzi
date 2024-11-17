#include "guiguzi/Logger.hpp"

#include "spdlog/spdlog.h"

#include "guiguzi/cv/FaceRecognition.hpp"

[[maybe_unused]] static void testOpenCV() {
    // double v = guiguzi::opencv_face_recognition("D:/tmp/face/1.png", "D:/tmp/face/1.png");
    // SPDLOG_DEBUG("1 = 1 = {:.6f}", v);
    // v = guiguzi::opencv_face_recognition("D:/tmp/face/1.png", "D:/tmp/face/2.png");
    // SPDLOG_DEBUG("1 = 2 = {:.6f}", v);
    // v = guiguzi::opencv_face_recognition("D:/tmp/face/1.png", "D:/tmp/face/3.png");
    // SPDLOG_DEBUG("1 = 3 = {:.6f}", v);
    // v = guiguzi::opencv_face_recognition("D:/tmp/face/2.png", "D:/tmp/face/3.png");
    // SPDLOG_DEBUG("2 = 3 = {:.6f}", v);
    guiguzi::opencv_face_recognition("D:/gitee/guiguzi/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg", "D:/tmp/face/c1.png");
    SPDLOG_DEBUG("====");
    guiguzi::opencv_face_recognition("D:/gitee/guiguzi/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg", "D:/tmp/face/c2.png");
    SPDLOG_DEBUG("====");
    guiguzi::opencv_face_recognition("D:/gitee/guiguzi/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg", "D:/tmp/face/c3.png");
}

[[maybe_unused]] static void testOnnx() {
    guiguzi::onnx_face_recognition();
}

/**
 * 人脸识别 = 检测 对齐 识别
 * https://blog.csdn.net/yangowen/article/details/128078481
 * https://blog.csdn.net/qq_42722197/article/details/121668671
 */

int main() {
    guiguzi::logger::init();
    testOnnx();
    // testOpenCV();
    guiguzi::logger::shutdown();
    return 0;
}
