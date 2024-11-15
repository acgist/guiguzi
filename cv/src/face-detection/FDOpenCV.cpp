#include "ggz/cv/FaceDetection.hpp"

#include <vector>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"

#include "opencv2/dnn.hpp"

static void dnn(const std::string& model, const std::string& path) {
    auto net = cv::dnn::readNetFromONNX(model);
    auto image = cv::imread(path);
    auto input = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(640, 640));
    // auto input = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    cv::imshow("FaceImage", input);
    cv::waitKey(0);
    cv::destroyWindow("FaceImage");
    net.setInput(input);
    auto pred = net.forward();
    cv::imshow("FaceImage", pred);
    cv::waitKey(0);
    cv::destroyWindow("FaceImage");
    cv::Mat ret(pred.size[1], pred.size[2], CV_32F, pred.ptr<float>());
    cv::imshow("FaceImage", ret);
    cv::waitKey(0);
    cv::destroyWindow("FaceImage");
}

static void simple(const std::string& model, const std::string& path) {
    cv::Mat image = cv::imread(path);
    cv::CascadeClassifier classifier;
    classifier.load(model);
    if(classifier.empty()) {
        return;
    }
    std::vector<cv::Rect> vector;
    classifier.detectMultiScale(image, vector);
    for(size_t i = 0; i < vector.size(); ++i) {
        cv::rectangle(image, vector[i].tl(), vector[i].br(), cv::Scalar(255, 0, 255), 3);
    }
    cv::imshow("FaceImage", image);
    cv::waitKey(0);
    cv::destroyWindow("FaceImage");
}

void ggz::opencv_face_detection(const std::string& model, const std::string& path) {
    dnn(model, path);
    // simple(model, path);
}
