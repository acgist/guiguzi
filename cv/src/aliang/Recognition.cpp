#include "Aliang.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

// 正脸矩阵
static std::vector<cv::Point> center_mat{
    cv::Point(38.2946, 51.6963),
    cv::Point(73.5318, 51.5014),
    cv::Point(56.0252, 71.7366),
    cv::Point(41.5493, 92.3655),
    cv::Point(70.7299, 92.2041)
};

static bool check_point(int w, int h, const cv::Rect& rect, const std::vector<cv::Point>& points) {
    if(rect.x + rect.width > w || rect.y + rect.height > h) {
        return false;
    }
    if(rect.x < 0 || rect.y < 0 || rect.width <= 0 || rect.height <= 0) {
        return false;
    }
    if(points.size() != 5) {
        return false;
    }
    for(auto& point : points) {
        if(point.x < 0 || point.y < 0) {
            return false;
        }
    }
    return true;
}

guiguzi::Recognition::Recognition(
    const std::string& face_model,    const char* face_logid,
    const std::string& feature_model, const char* feature_logid,
    float threshold, float confidenceThreshold, float iouThreshold
) : threshold(threshold) {
    std::vector<std::string> classes{ "face" };
    this->faceModel = std::make_unique<guiguzi::OnnxRuntime>(640, classes, confidenceThreshold, iouThreshold);
    this->faceModel->createSession(face_model, face_logid);
    this->featureModel = std::make_unique<guiguzi::OnnxRuntime>(112);
    this->featureModel->createSession(feature_model, feature_logid);
}

bool guiguzi::Recognition::extract(cv::Mat& image, std::vector<cv::Rect>& boxes, std::vector<cv::Point>& points) {
    float scale = 1.0F;
    cv::Mat output = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    float* blob = guiguzi::formatImage(this->faceModel->wh, image, output, scale);
    this->faceModel->run(blob, scale, boxes, points);
    return true;
}

bool guiguzi::Recognition::center(cv::Mat& image, std::vector<cv::Point>& points) {
    int min = std::min(image.cols, image.rows);
    cv::Rect rect(0, 0, min, min);
    if(min == image.cols) {
        rect.y = (image.rows - min) / 2;
    } else {
        rect.x = (image.cols - min) / 2;
    }
    for(auto& point : points) {
        point.x -= rect.x;
        point.y -= rect.y;
    }
    if(!check_point(image.cols, image.rows, rect, points)) {
        return false;
    }
    image = image(rect);
    float scale = 1.0F * this->featureModel->wh / min;
    for(auto& point : points) {
        point.x *= scale;
        point.y *= scale;
    }
    cv::resize(image, image, cv::Size(this->featureModel->wh, this->featureModel->wh));
    auto M = cv::estimateAffinePartial2D(points, center_mat);
    cv::warpAffine(image, image, M, image.size());
    return true;
}

void guiguzi::Recognition::feature(cv::Mat& image, std::vector<float>& feature) {
    cv::Mat output;
    cv::dnn::blobFromImage(image, output, 1.0 / 255.0, cv::Size(this->featureModel->wh, this->featureModel->wh), cv::Scalar(), true, false);
    float* blob = reinterpret_cast<float*>(output.data);
    int64_t signalResultNum;
    int64_t strideNum;
    this->featureModel->run(blob, feature, signalResultNum, strideNum);
    cv::normalize(feature, feature);
}

double guiguzi::Recognition::compare(const std::vector<float>& source, const std::vector<float>& target) {
    double v = 0.0;
    for(int i = 0; i < source.size(); ++i) {
        v += std::pow((source[i] - target[i]), 2);
    }
    return v;
}

void guiguzi::Recognition::storage(const std::string& name, std::vector<cv::Mat>& images) {
    if(images.empty()) {
        return;
    }
    std::vector<std::vector<float>> features;
    for(auto& image : images) {
        std::vector<cv::Rect> boxes;
        std::vector<cv::Point> points;
        this->extract(image, boxes, points);
        if(boxes.empty()) {
            SPDLOG_INFO("没有人脸数据：{}", name);
            continue;
        }
        if(boxes.size() != 1) {
            SPDLOG_INFO("多张人脸数据：{}", name);
            continue;
        }
        auto rect = boxes[0];
        for(auto& point : points) {
            point.x -= rect.x;
            point.y -= rect.y;
        }
        if(!check_point(image.cols, image.rows, rect, points)) {
            SPDLOG_INFO("人脸数据错误：{}", name);
            continue;
        }
        image = image(rect);
        if(!this->center(image, points)) {
            SPDLOG_INFO("人脸数据错误：{}", name);
            continue;
        }
        std::vector<float> feature;
        this->feature(image, feature);
        features.push_back(feature);
    }
    this->features.emplace(name, features);
    SPDLOG_DEBUG("人脸特征：{} - {}", name, features.size());
}

void guiguzi::Recognition::storage(const std::string& name, const std::vector<std::string>& images) {
    std::vector<cv::Mat> mats;
    for(const auto& image : images) {
        mats.push_back(std::move(cv::imread(image)));
    }
    this->storage(name, mats);
}

std::pair<std::string, double> guiguzi::Recognition::recognition(cv::Mat& image) {
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point> points;
    this->extract(image, boxes, points);
    if(boxes.empty()) {
        return {};
    }
    // 描点
    // auto point = points.begin();
    // for(const auto& rect : boxes) {
    //     cv::rectangle(image, rect, cv::Scalar{ 255, 255, 0 });
    //     cv::circle(image, *point, 2, cv::Scalar{ 255, 255, 0 });
    //     ++point;
    //     cv::circle(image, *point, 2, cv::Scalar{ 255, 255, 0 });
    //     ++point;
    //     cv::circle(image, *point, 2, cv::Scalar{ 255, 255, 0 });
    //     ++point;
    //     cv::circle(image, *point, 2, cv::Scalar{ 255, 255, 0 });
    //     ++point;
    //     cv::circle(image, *point, 2, cv::Scalar{ 255, 255, 0 });
    //     ++point;
    // }
    auto rect = boxes[0];
    for(auto& point : points) {
        point.x -= rect.x;
        point.y -= rect.y;
    }
    if(!check_point(image.cols, image.rows, rect, points)) {
        return {};
    }
    image = image(rect);
    if(!this->center(image, points)) {
        return {};
    }
    std::vector<float> feature;
    this->feature(image, feature);
    std::string retk;
    double retv = 100.0;
    for(const auto&[ k, v ] : this->features) {
        for(const auto& f : v) {
            double same = this->compare(feature, f);
            if(same < retv) {
                retv = same;
                retk = k;
            }
        }
    }
    if(retv > this->threshold || retk.empty()) {
        return {};
    }
    return std::make_pair<>(retk, retv);
}
