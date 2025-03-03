#include "opencv2/opencv.hpp"

int main() {
    cv::Mat source{ cv::imread("D:/tmp/xxv.jpg") };
    cv::imshow("cv", source);
    cv::waitKey(0);
    cv::Mat target{ cv::imread("D:/tmp/xxc.png") };
    cv::imshow("cv", target);
    cv::waitKey(0);
    cv::Mat range{ source, cv::Range(0, target.rows), cv::Range(0, target.cols) };
    cv::imshow("cv", range);
    cv::waitKey(0);
    target.copyTo(range);
    cv::imshow("cv", source);
    cv::waitKey(0);
    cv::destroyAllWindows();
    // cv::Mat_<int>(2, 2) << 1, 2, 3;
    return 0;
}