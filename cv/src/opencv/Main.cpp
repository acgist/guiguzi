#include "guiguzi/Logger.hpp"

#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

[[maybe_unused]] static void testBlobFromImage() {
    auto image = cv::imread("D:/tmp/F4.jpg");
    cv::imshow("image", image);
    int col = image.cols;
    int row = image.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    image.copyTo(result(cv::Rect(0, 0, col, row)));
    cv::imshow("result", result);
    cv::Mat target;
    cv::dnn::blobFromImage(result, target, 1.0, cv::Size(640, 640), cv::Scalar(), true, false);
    // cv::dnn::blobFromImage(image, target, 1.0 / 255, cv::Size(640, 640), cv::Scalar(), true, false);
    // cv::imshow("target", target);
}

int main() {
    guiguzi::logger::init();
    testBlobFromImage();
    cv::waitKey(0);
    guiguzi::logger::shutdown();
    return 0;
}