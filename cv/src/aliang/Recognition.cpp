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

guiguzi::Recognition::Recognition(
    const std::string& face_model,    const char* face_logid,
    const std::string& feature_model, const char* feature_logid,
    float threshold, float confidenceThreshold, float iouThreshold
) : threshold(threshold) {
    std::vector<std::string> classes{ "face" };
    this->faceModel = std::make_unique<guiguzi::OnnxRuntime>(640, face_logid, classes, confidenceThreshold, iouThreshold);
    this->faceModel->createSession(face_model);
    this->featureModel = std::make_unique<guiguzi::OnnxRuntime>(112, feature_logid);
    this->featureModel->createSession(feature_model);
}

bool guiguzi::Recognition::extract(cv::Mat& image, std::vector<cv::Rect>& boxes, std::vector<cv::Point>& points) {
    float scale;
    cv::Mat output;
    float* blob = guiguzi::formatBlob(this->faceModel->wh, image, output, scale);
    this->faceModel->run(blob, scale, boxes, points);
    if(boxes.empty()) {
        SPDLOG_DEBUG("没有检测到人脸数据");
        return false;
    }
    if(boxes.size() != 1) {
        SPDLOG_DEBUG("检测到多张人脸数据");
        return false;
    }
    // 修正参数
    for(auto& rect : boxes) {
        guiguzi::fixRect(image, rect);
    }
    return true;
}

void guiguzi::Recognition::center(cv::Mat& image, std::vector<cv::Point>& points) {
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
    image = image(rect);
    float scale = 1.0F * this->featureModel->wh / min;
    for(auto& point : points) {
        point.x *= scale;
        point.y *= scale;
    }
    cv::resize(image, image, cv::Size(this->featureModel->wh, this->featureModel->wh));
    auto M = cv::estimateAffinePartial2D(points, center_mat);
    cv::warpAffine(image, image, M, image.size());
}

void guiguzi::Recognition::feature(cv::Mat& image, std::vector<float>& feature) {
    cv::Mat output;
    cv::dnn::blobFromImage(image, output, 1.0 / 255.0, cv::Size(this->featureModel->wh, this->featureModel->wh), cv::Scalar(), true, false);
    float* blob = reinterpret_cast<float*>(output.data);
    std::vector<int64_t> dims;
    this->featureModel->run(blob, feature, dims);
    cv::normalize(feature, feature);
}

inline double guiguzi::Recognition::compare(const std::vector<float>& source, const std::vector<float>& target) {
    double ret = 0.0;
    for(int i = 0; i < source.size(); ++i) {
        ret += std::pow((source[i] - target[i]), 2);
    }
    return ret;
}

void guiguzi::Recognition::storage(const std::string& name, std::vector<cv::Mat>& images) {
    if(images.empty()) {
        return;
    }
    std::vector<std::vector<float>> features;
    for(auto& image : images) {
        std::vector<cv::Rect> boxes;
        std::vector<cv::Point> points;
        if(!this->extract(image, boxes, points)) {
            SPDLOG_WARN("人脸录入失败：{}", name);
            continue;
        }
        auto rect = boxes[0];
        for(auto& point : points) {
            point.x -= rect.x;
            point.y -= rect.y;
        }
        image = image(rect);
        this->center(image, points);
        std::vector<float> feature;
        this->feature(image, feature);
        features.push_back(feature);
    }
    this->features.emplace(name, features);
    SPDLOG_INFO("录入人脸特征：{} - {}", name, features.size());
}

void guiguzi::Recognition::storage(const std::string& name, const std::vector<std::string>& images) {
    std::vector<cv::Mat> mats;
    for(const auto& image : images) {
        mats.push_back(std::move(cv::imread(image)));
    }
    this->storage(name, mats);
}

std::pair<std::string, double> guiguzi::Recognition::recognition(cv::Mat& image, std::vector<cv::Point>& points_ori) {
    std::vector<cv::Rect>  boxes;
    std::vector<cv::Point> points;
    if(!this->extract(image, boxes, points)) {
        SPDLOG_DEBUG("人脸识别失败");
        return {};
    }
    auto rect = boxes[0];
    for(auto& point : points) {
        points_ori.push_back(point);
        point.x -= rect.x;
        point.y -= rect.y;
    }
    image = image(rect);
    this->center(image, points);
    std::vector<float> feature;
    this->feature(image, feature);
    // 比较特征
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
