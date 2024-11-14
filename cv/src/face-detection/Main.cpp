#include "ggz/Logger.hpp"

#include "spdlog/spdlog.h"

#include "ggz/cv/FaceDetection.hpp"

[[maybe_unused]] static void testOpenCV() {
    ggz::opencv_face_detection("D:/gitee/ggz/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/F4.jpg");
    // ggz::opencv_face_detection("D:/gitee/ggz/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml", "D:/tmp/head.jpg");
}

int main() {
    ggz::logger::init();
    testOpenCV();
    ggz::logger::shutdown();
    return 0;
}
