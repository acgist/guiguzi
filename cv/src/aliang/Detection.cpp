#include "Aliang.hpp"

#include "opencv2/opencv.hpp"

guiguzi::Detection::Detection(
    const std::string& model,
    const char* logid,
    const std::vector<std::string>& classes,
    float confidenceThreshold,
    float iouThreshold
) : classes(classes) {
    this->model = std::make_unique<guiguzi::OnnxRuntime>(640, classes, confidenceThreshold, iouThreshold);
    this->model->createSession(model, logid);
}

void guiguzi::Detection::detection(
    const cv::Mat& image,              // 图片
    std::vector<int>& class_ids,       // 类型
    std::vector<float>& confidences,   // 置信度
    std::vector<cv::Rect>& boxes // 框
) {
    float scale = 1.0F;
    cv::Mat output;
    float* blob = guiguzi::formatImage(this->model->wh, image, output, scale);
    this->model->run(blob, scale, class_ids, confidences, boxes);
}

void guiguzi::Detection::detection(cv::Mat& image, Recognition& recognition) {
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    this->detection(image, class_ids, confidences, boxes);
    auto class_id   = class_ids.begin();
    auto confidence = confidences.begin();
    for(const auto& rect : boxes) {
        cv::Mat face = image(rect).clone();
        auto person = recognition.recognition(face);
        if(!person.first.empty()) {
            cv::putText(
                image,
                std::to_string(person.second),
                cv::Point(rect.x, rect.y + 40),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(0, 0, 0),
                2
            );
            cv::putText(
                image,
                person.first,
                cv::Point(rect.x, rect.y + 60),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(0, 0, 0),
                2
            );
        }
        cv::rectangle(image, rect, cv::Scalar{ 255, 0, 0 });
        cv::putText(
            image,
            std::to_string(*confidence),
            cv::Point(rect.x, rect.y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 0, 0),
            2
        );
        cv::putText(
            image,
            this->classes[*class_id],
            cv::Point(rect.x, rect.y + 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 0, 0),
            2
        );
        ++class_id;
        ++confidence;
    }
}
