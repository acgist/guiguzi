#include "ggz/Logger.hpp"

#include "opencv2/opencv.hpp"

static void testCpature() {
    cv::CascadeClassifier classifier;
    classifier.load("D:/gitee/ggz/deps/opencv/etc/haarcascades/haarcascade_frontalface_default.xml");
    if(classifier.empty()) {
        return;
    }
    cv::VideoCapture vc(0);
    cv::Mat frame;
    vc >> frame;
    while(!frame.empty()) {
        std::vector<cv::Rect> vector;
        classifier.detectMultiScale(frame, vector);
        for(size_t i = 0; i < vector.size(); ++i) {
            cv::rectangle(frame, vector[i].tl(), vector[i].br(), cv::Scalar(255, 0, 255), 3);
        }
        // cv::putText(frame, "acgist", cv::Point(100, 100), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        cv::putText(frame, "碧螺萧萧", cv::Point(100, 100), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        cv::imshow("frame", frame);
        if(cv::waitKey(1) == 27) {
            // ESC
            break;
        }
        vc >> frame;
    }
}

int main() {
    ggz::logger::init();
    testCpature();
    ggz::logger::shutdown();
    return 0;
}