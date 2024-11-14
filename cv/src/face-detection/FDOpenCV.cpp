#include "ggz/cv/FaceDetection.hpp"

#include <vector>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"

void ggz::opencv_face_detection(const std::string& model, const std::string& path) {
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
