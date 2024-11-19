#include "guiguzi/Logger.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

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

[[maybe_unused]] static void testFaceFeature() {
    cv::Mat face1 = cv::imread("D:/tmp/face/1.png");
    cv::Mat face2 = cv::imread("D:/tmp/face/2.png");
    cv::Mat face3 = cv::imread("D:/tmp/face/3.png");
    cv::Mat facea = cv::imread("D:/tmp/face/c1.png");
    cv::Mat faceb = cv::imread("D:/tmp/face/c2.png");
    cv::Mat facec = cv::imread("D:/tmp/face/c3.png");
    auto f1 = guiguzi::onnx_face_feature(face1);
    cv::waitKey(0);
    auto f2 = guiguzi::onnx_face_feature(face2);
    cv::waitKey(0);
    auto f3 = guiguzi::onnx_face_feature(face3);
    cv::waitKey(0);
    auto fa = guiguzi::onnx_face_feature(facea);
    cv::waitKey(0);
    auto fb = guiguzi::onnx_face_feature(faceb);
    cv::waitKey(0);
    auto fc = guiguzi::onnx_face_feature(facec);
    cv::waitKey(0);
    double d_1_2 = guiguzi::onnx_face_feature_diff(f1, f2);
    double d_1_3 = guiguzi::onnx_face_feature_diff(f1, f3);
    double d_2_3 = guiguzi::onnx_face_feature_diff(f2, f3);
    double d_a_b = guiguzi::onnx_face_feature_diff(fa, fb);
    double d_a_c = guiguzi::onnx_face_feature_diff(fa, fc);
    double d_b_c = guiguzi::onnx_face_feature_diff(fb, fc);
    double d_1_a = guiguzi::onnx_face_feature_diff(f1, fa);
    SPDLOG_DEBUG("d_1_2 = {}", d_1_2);
    SPDLOG_DEBUG("d_1_3 = {}", d_1_3);
    SPDLOG_DEBUG("d_2_3 = {}", d_2_3);
    SPDLOG_DEBUG("d_a_b = {}", d_a_b);
    SPDLOG_DEBUG("d_a_c = {}", d_a_c);
    SPDLOG_DEBUG("d_b_c = {}", d_b_c);
    SPDLOG_DEBUG("d_1_a = {}", d_1_a);
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
    // testFaceFeature();
    guiguzi::logger::shutdown();
    return 0;
}
