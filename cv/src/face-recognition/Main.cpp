#include "ggz/Logger.hpp"

#include "spdlog/spdlog.h"

#include "ggz/cv/FaceRecognition.hpp"

#include <iostream>

[[maybe_unused]] static void testOpenCV() {
    // double v = ggz::opencv_face_recognition("D:/tmp/face/1.png", "D:/tmp/face/1.png");
    // SPDLOG_DEBUG("1 = 1 = {:.6f}", v);
    // v = ggz::opencv_face_recognition("D:/tmp/face/1.png", "D:/tmp/face/2.png");
    // SPDLOG_DEBUG("1 = 2 = {:.6f}", v);
    // v = ggz::opencv_face_recognition("D:/tmp/face/1.png", "D:/tmp/face/3.png");
    // SPDLOG_DEBUG("1 = 3 = {:.6f}", v);
    // v = ggz::opencv_face_recognition("D:/tmp/face/2.png", "D:/tmp/face/3.png");
    // SPDLOG_DEBUG("2 = 3 = {:.6f}", v);
    ggz::opencv_face_recognition("D:/gitee/ggz/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg", "D:/tmp/face/c1.png");
    SPDLOG_DEBUG("====");
    ggz::opencv_face_recognition("D:/gitee/ggz/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg", "D:/tmp/face/c2.png");
    SPDLOG_DEBUG("====");
    ggz::opencv_face_recognition("D:/gitee/ggz/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg", "D:/tmp/face/c3.png");
}

int main() {
    ggz::logger::init();
    testOpenCV();
    ggz::logger::shutdown();
    return 0;
}
