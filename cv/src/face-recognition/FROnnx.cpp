/**
 * 人脸提取
 * yoloface_8n
 * https://github.com/akanametov/yolo-face
 * 人脸识别
 * arcface_w600k_r50
 */

#include "guiguzi/cv/FaceRecognition.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

#include "onnxruntime_cxx_api.h"

static Ort::Env* env;
static Ort::Session* session;
static Ort::RunOptions options{ nullptr };

static std::vector<const char*> inputNodeNames;
static std::vector<const char*> outputNodeNames;

static float resizeScales;
static std::vector<int> imgSize{ 640, 640 };
static cv::Size2f modelShape = cv::Size(640, 640);

static void releaseSession() {
    delete env;
    env = nullptr;
    delete session;
    session = nullptr;
}

static std::vector<cv::Rect> postProcess(float* blob, std::vector<int64_t>& inputNodeDims, std::vector<cv::Point>& points) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<float*>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        blob,
        3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size()
    );
    auto outputTensor = session->Run(
        options,
        inputNodeNames.data(),
        &inputTensor, 1,
        outputNodeNames.data(),
        outputNodeNames.size()
    );
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<float*>::type>();
    // TODO: 多个结果是否需要转置
    // 转换
    int signalResultNum = outputNodeDims[1]; // 20
    int strideNum       = outputNodeDims[2]; // 8400
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat rawData;
    rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
    rawData = rawData.t();
    float* data = (float*) rawData.data;
    std::vector<cv::Point> pointsTmp;
    for (int i = 0; i < strideNum; ++i) {
        float* classesScores = data + 4;
        // 20 - 4
        cv::Point class_id;
        float maxClassScore = data[4];
        if (maxClassScore > 0.6) { // 置信度
            // for(int x = 0; x < signalResultNum; ++x) {
            //     SPDLOG_DEBUG("========== {}", data[x]);
            // }
            // SPDLOG_DEBUG("----------------{}", maxClassScore);
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * resizeScales);
            int top  = int((y - 0.5 * h) * resizeScales);

            int width  = int(w * resizeScales);
            int height = int(h * resizeScales);

            pointsTmp.push_back(cv::Point((data[5]  * resizeScales), (data[6]  * resizeScales)));
            pointsTmp.push_back(cv::Point((data[8]  * resizeScales), (data[9]  * resizeScales)));
            pointsTmp.push_back(cv::Point((data[11] * resizeScales), (data[12] * resizeScales)));
            pointsTmp.push_back(cv::Point((data[14] * resizeScales), (data[15] * resizeScales)));
            pointsTmp.push_back(cv::Point((data[17] * resizeScales), (data[18] * resizeScales)));

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += signalResultNum;
    }
    std::vector<int> nmsResult;
    std::vector<cv::Rect> rest;
    cv::dnn::NMSBoxes(boxes, confidences, 0.6, 0.0, nmsResult);
    for(const auto& i : nmsResult) {
        rest.push_back(boxes[i]);
        points.push_back(pointsTmp[5 * i + 0]);
        points.push_back(pointsTmp[5 * i + 1]);
        points.push_back(pointsTmp[5 * i + 2]);
        points.push_back(pointsTmp[5 * i + 3]);
        points.push_back(pointsTmp[5 * i + 4]);
    }
    return rest;
}

static cv::Mat formatToSquare(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    resizeScales = 1.0F * _max / 640;
    return result;
}

static std::vector<float> preProcess(cv::Mat& source, cv::Mat& target) {
    target = source.clone();

    // cv::cvtColor(target, target, cv::COLOR_BGR2RGB);
    // if (source.cols >= source.rows) {
    //     resizeScales = source.cols / (float) imgSize.at(0);
    //     cv::resize(target, target, cv::Size(imgSize.at(0), int(source.rows / resizeScales)));
    // } else {
    //     resizeScales = source.rows / (float) imgSize.at(0);
    //     cv::resize(target, target, cv::Size(int(source.cols / resizeScales), imgSize.at(1)));
    // }
    // cv::Mat tempImg = cv::Mat::zeros(imgSize.at(0), imgSize.at(1), CV_8UC3);
    // target.copyTo(tempImg(cv::Rect(0, 0, target.cols, target.rows)));
    // target = tempImg;

    // float* blob   = new float[target.total() * 3];
    // int channels  = target.channels();
    // int imgHeight = target.rows;
    // int imgWidth  = target.cols;
    // for (int c = 0; c < channels; c++)
    // {
    //     for (int h = 0; h < imgHeight; h++)
    //     {
    //         for (int w = 0; w < imgWidth; w++)
    //         {
    //             blob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<float*>::type((target.at<cv::Vec3b>(h, w)[c]) / 255.0f);
    //         }
    //     }
    // }

    cv::Mat input;
    target = formatToSquare(target);
    cv::dnn::blobFromImage(target, input, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    float* blob = reinterpret_cast<float*>(input.data);
    std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    // postProcess(reinterpret_cast<float*>(input.data), inputNodeDims);
    std::vector<cv::Point> points;
    std::vector<cv::Rect> boxs = postProcess(blob, inputNodeDims, points);
    cv::Mat face;
    for(const auto& rect : boxs) {
        // cv::rectangle(target, rect, cv::Scalar{ 255, 0, 0 });
        cv::rectangle(source, rect, cv::Scalar{ 255, 0, 0 });
        face = source(rect).clone();
        auto min = std::min(face.rows, face.cols);
        if(min == face.rows) {
            int pos = (face.cols - min) / 2;
            double scale = 1.0 * min / 112;
            for(auto& p : points) {
                p.x = (p.x - rect.x - pos) / scale;
                p.y = (p.y - rect.y) / scale;
            }
        } else {
            int pos = (face.rows - min) / 2;
            double scale = 1.0 * min / 112;
            for(auto& p : points) {
                p.x = (p.x - rect.x) / scale;
                p.y = (p.y - rect.y - pos) / scale;
            }
        }
        cv::Rect rect((face.cols - min) / 2, (face.rows - min) / 2, min, min);
        face = face(rect);
        // cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        // cv::resize(input, input, cv::Size(112, 112));
        cv::resize(face, face, cv::Size(112, 112));
        // cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 255), 3);
    }
    cv::imshow("xx", face);
    // SPDLOG_DEBUG("[ {}, {}, {}, {}, {}, {}, {}, {}, {}, {} ]",
    //     points[0].x, points[0].y,
    //     points[1].x, points[1].y,
    //     points[2].x, points[2].y,
    //     points[3].x, points[3].y,
    //     points[4].x, points[4].y
    // );
    int i = 0;
    // for(const auto& point : points) {
    //     cv::putText(
    //         source,
    //         std::to_string(i),
    //         point,
    //         cv::FONT_HERSHEY_SIMPLEX,
    //         1.0,
    //         cv::Scalar(0, 0, 0),
    //         2
    //     );
    //     i++;
    // }
    guiguzi::onnx_face_feature_center(face, points);
    auto feature = guiguzi::onnx_face_feature(face);
    // cv::waitKey(0);
    // delete[] blob;
    return feature;
}

static void createSession() {
    if(env) {
        delete env;
        env = nullptr;
    }
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
    Ort::SessionOptions sessionOption;
    sessionOption.SetLogSeverityLevel(3);
    sessionOption.SetIntraOpNumThreads(1);
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if(session) {
        delete session;
        session = nullptr;
    }
    // session = new Ort::Session(*env, L"D:/tmp/face/yolov8n-face.onnx", sessionOption);
    session = new Ort::Session(*env, L"D:/tmp/face/yoloface_8n.onnx", sessionOption);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++) {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
        SPDLOG_DEBUG("输入参数：{}", temp_buf);
    }
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++) {
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
        SPDLOG_DEBUG("输出参数：{}", temp_buf);
    }
    options = Ort::RunOptions{ nullptr };
}

void guiguzi::onnx_face_recognition() {
    createSession();
    // cv::Mat frame;
    // cv::Mat output;
    // cv::VideoCapture camera(0);
    // while(true) {
    //     camera >> frame;
    //     preProcess(frame, output);
    //     cv::imshow("face", frame);
    //     if(cv::waitKey(1) == 27) {
    //         break;
    //     }
    // }
    // camera.release();

    // cv::Mat frame = cv::imread("D:/tmp/F4.jpg");

    cv::Mat frame1 = cv::imread("D:/tmp/face/y1.jpg");
    cv::Mat output1;
    auto v1 = preProcess(frame1, output1);
    // cv::Mat frame2 = cv::imread("D:/tmp/face/y2.jpg");
    cv::Mat frame2 = cv::imread("D:/tmp/face/y4.jpg");
    cv::Mat output2;
    auto v2 = preProcess(frame2, output2);
    // cv::Mat frame3 = cv::imread("D:/tmp/face/y3.jpg");
    cv::Mat frame3 = cv::imread("D:/tmp/face/y5.jpg");
    cv::Mat output3;
    auto v3 = preProcess(frame3, output3);

    // cv::Mat frame1 = cv::imread("D:/tmp/face/1.png");
    // cv::Mat output1;
    // auto v1 = preProcess(frame1, output1);
    // cv::Mat frame2 = cv::imread("D:/tmp/face/2.png");
    // cv::Mat output2;
    // auto v2 = preProcess(frame2, output2);
    // cv::Mat frame3 = cv::imread("D:/tmp/face/3.png");
    // cv::Mat output3;
    // auto v3 = preProcess(frame3, output3);

    cv::Mat cframe1 = cv::imread("D:/tmp/face/c1.png");
    cv::Mat coutput1;
    auto cv1 = preProcess(cframe1, coutput1);
    cv::Mat cframe2 = cv::imread("D:/tmp/face/c2.png");
    cv::Mat coutput2;
    auto cv2 = preProcess(cframe2, coutput2);
    cv::Mat cframe3 = cv::imread("D:/tmp/face/c3.png");
    cv::Mat coutput3;
    auto cv3 = preProcess(cframe3, coutput3);

    auto d12 = guiguzi::onnx_face_feature_diff(v1, v2);
    SPDLOG_DEBUG("{}", d12);
    auto d23 = guiguzi::onnx_face_feature_diff(v2, v3);
    SPDLOG_DEBUG("{}", d23);
    auto d13 = guiguzi::onnx_face_feature_diff(v1, v3);
    SPDLOG_DEBUG("{}", d13);

    auto cd12 = guiguzi::onnx_face_feature_diff(cv1, cv2);
    SPDLOG_DEBUG("{}", cd12);
    auto cd23 = guiguzi::onnx_face_feature_diff(cv2, cv3);
    SPDLOG_DEBUG("{}", cd23);
    auto cd13 = guiguzi::onnx_face_feature_diff(cv1, cv3);
    SPDLOG_DEBUG("{}", cd13);

    auto cv = guiguzi::onnx_face_feature_diff(v1, cv3);
    SPDLOG_DEBUG("{}", cv);

    // cv::imshow("frame1", frame1);
    // cv::imshow("output1", output1);
    cv::waitKey(0);
    releaseSession();
}
