#include "guiguzi/Logger.hpp"

#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

#include "spdlog/spdlog.h"

[[maybe_unused]] static void createWindow() {
    cv::namedWindow("ceshi", cv::WINDOW_NORMAL);
    // cv::resizeWindow
    cv::imshow("ceshi", 0);
}

[[maybe_unused]] static void testBlobFromImage() {
    auto image = cv::imread("D:/tmp/F4.jpg");
    cv::imshow("image", image.clone());
    int col = image.cols;
    int row = image.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    image.copyTo(result(cv::Rect(0, 0, col, row)));
    cv::imshow("result", result.clone());
    cv::Mat target;
    // cv::dnn::blobFromImage(result, target, 1.0F / 255, cv::Size(640, 640), cv::Scalar(), true, false);
    // cv::dnn::blobFromImage(result, target, 1.0F, cv::Size(640, 640), cv::Scalar(), true, false);
    cv::dnn::blobFromImage(result, target, 1.0F / 255, cv::Size(640, 640), cv::Scalar(), true, false);
    // float* sa = reinterpret_cast<float*>(target.data);
    float* array = new float[3 * 640 * 640];
    // std::copy(sa, sa + 3 * 640 * 640, array);
    cv::Mat targetv(640, 640, CV_32FC3, target.data);
    cv::imshow("targetv", targetv.clone());

    float resizeScales;
    cv::Mat source = image.clone();
    target = image.clone();
    std::vector<int> imgSize{ 640, 640 };
    cv::cvtColor(target, target, cv::COLOR_BGR2RGB);
    if (source.cols >= source.rows) {
        resizeScales = source.cols / (float) imgSize.at(0);
        cv::resize(target, target, cv::Size(imgSize.at(0), int(source.rows / resizeScales)));
    } else {
        resizeScales = source.rows / (float) imgSize.at(0);
        cv::resize(target, target, cv::Size(int(source.cols / resizeScales), imgSize.at(1)));
    }
    cv::Mat tempImg = cv::Mat::zeros(imgSize.at(0), imgSize.at(1), CV_8UC3);
    target.copyTo(tempImg(cv::Rect(0, 0, target.cols, target.rows)));
    target = tempImg;
    cv::imshow("targetx", target);

    auto total = target.total();
    float* blob   = new float[target.total() * 3];
    int channels  = target.channels();
    int imgHeight = target.rows;
    int imgWidth  = target.cols;
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                // blob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<float*>::type((target.at<cv::Vec3b>(h, w)[c]) / 1.0f);
                blob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<float*>::type((target.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    cv::Mat targetex(640, 640, CV_32FC3, blob);
    cv::imshow("targetex", targetex.clone());

    bool ret = std::equal(blob, blob + target.total() * 3, array, [](const auto& a, const auto& b) {
        if(a - b != 0.0F) {
            // SPDLOG_DEBUG("diff = {}", a - b);
            if(std::abs(a - b) < 0.001) {
                return true;
            } else {
                return false;
            }
        } else {
            return true;
        }
    });
    SPDLOG_DEBUG("all equal : {}", ret);
    delete[] array;
}

[[maybe_unused]] static void testCamera() {
    cv::VideoCapture camera(0);
    cv::Mat frame;
    while(camera.read(frame)) {
        cv::imshow("camera", frame);
        auto key = cv::waitKey(1);
        if(key == 'q') {
            break;
        }
    }
    camera.release();
    cv::destroyAllWindows();
}

// constexpr int timeDuration() {
//     return 1000 / 25;
// }

constexpr static int timeDuration = 1000 / 25;

[[maybe_unused]] static void testFile() {
    cv::VideoCapture file("D:/tmp/video-copy.mp4");
    cv::Mat frame;
    while(file.read(frame)) {
        cv::imshow("camera", frame);
        auto key = cv::waitKey(timeDuration);
        if(key == 'q') {
            break;
        }
    }
    file.release();
    cv::destroyAllWindows();
}

[[maybe_unused]] static void testFileWrite() {
    cv::VideoCapture capture(0);
    cv::Mat frame;
    cv::VideoWriter writer(
        "D:/tmp/camera.mp4",
        // vp90
        // mjpg
        // xvid
        // h264/avc1
        // h265/hevc
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        25,
        cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    while(capture.read(frame)) {
        writer.write(frame);
        cv::imshow("camera", frame);
        auto key = cv::waitKey(1000 / 25);
        if(key == 'q') {
            break;
        }
    }
    writer.release();
    capture.release();
}

int main() {
    guiguzi::logger::init();
    // createWindow();
    // testBlobFromImage();
    // testCamera();
    // testFile();
    testFileWrite();
    cv::waitKey(0);
    // while(true) {
    //     auto key = cv::waitKey(0);
    //     if(key == 27) {
    //     // if(key == 'q') {
    //         break;
    //     }
    // }
    guiguzi::logger::shutdown();
    return 0;
}