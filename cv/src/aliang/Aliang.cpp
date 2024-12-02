#include "Aliang.hpp"

#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

float* guiguzi::formatImage(int wh, const cv::Mat& source, cv::Mat& target, float& scale) {
    // 计算比例
    const int col = source.cols;
    const int row = source.rows;
    const int max = MAX(col, row);
    scale = 1.0F * max / wh;
    // 复制图片
    cv::Mat result = cv::Mat::zeros(max, max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    // 格式转换
    cv::dnn::blobFromImage(result, target, 1.0 / 255.0, cv::Size(wh, wh), cv::Scalar(), true, false);
    return reinterpret_cast<float*>(target.data);
}
