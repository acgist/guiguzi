#include "guiguzi/cv/FaceRecognition.hpp"

#include <vector>

#include "spdlog/spdlog.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc/types_c.h"

static double opencv_face_recognition(const cv::Mat& source, const cv::Mat& target) {
    cv::Mat sourceNew;
    cv::Mat targetNew;
    cv::Mat sourceHist;
    cv::Mat targetHist;
    cv::Mat sourceGray;
    cv::Mat targetGray;
    cv::resize(source, sourceNew, cv::Size(128, 128));
    cv::resize(target, targetNew, cv::Size(128, 128));
    // 灰度
    cv::cvtColor(sourceNew, sourceGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(targetNew, targetGray, cv::COLOR_BGR2GRAY);
    // 直方图大小：越大匹配越精确越慢
    int histSize = 255;
    // 颜色范围
    float range[] = { 0, 255 } ;
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    // cv::imshow("FaceImage", sourceGray);
    // cv::waitKey(0);
    // cv::destroyWindow("FaceImage");
    // cv::imshow("FaceImage", targetGray);
    // cv::waitKey(0);
    // cv::destroyWindow("FaceImage");
    cv::calcHist(&sourceGray, 1, 0, cv::Mat(), sourceHist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&targetGray, 1, 0, cv::Mat(), targetHist, 1, &histSize, &histRange, uniform, accumulate);
    return cv::compareHist(sourceHist, targetHist, CV_COMP_CORREL);
}

double guiguzi::opencv_face_recognition(const std::string& source, const std::string& target) {
    return ::opencv_face_recognition(cv::imread(source), cv::imread(target));
}

void guiguzi::opencv_face_recognition(const std::string& model, const std::string& path, const std::string& face) {
    cv::Mat image = cv::imread(path);
    cv::Mat target = cv::imread(face);
    cv::CascadeClassifier classifier;
    classifier.load(model);
    if(classifier.empty()) {
        return;
    }
    std::vector<cv::Rect> vector;
    classifier.detectMultiScale(target, vector);
    cv::Mat diff;
    for(size_t i = 0; i < vector.size(); ++i) {
        cv::Rect crop(vector[i].tl(), vector[i].br());
        diff = image(crop);
    }
    if(diff.empty()) {
        return;
    }
    vector.clear();
    classifier.detectMultiScale(image, vector);
    double last = 0.0;
    cv::Point tl;
    cv::Point br;
    for(size_t i = 0; i < vector.size(); ++i) {
        cv::Rect crop(vector[i].tl(), vector[i].br());
        cv::Mat copy = image(crop);
        double now = ::opencv_face_recognition(copy, diff);
        SPDLOG_DEBUG("当前相似度：{}", now);
        if(now > last) {
            last = now;
            tl = vector[i].tl();
            br = vector[i].br();
        }
    }
    cv::rectangle(image, tl, br, cv::Scalar(255, 0, 255), 3);
    cv::imshow("FaceImage", image);
    cv::waitKey(0);
    cv::destroyWindow("FaceImage");
}
