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

static void postProcess(float* blob, std::vector<int64_t>& inputNodeDims) {
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
    for (int i = 0; i < strideNum; ++i) {
        float* classesScores = data + 4;
        // 20 - 4
        cv::Point class_id;
        float maxClassScore = data[4];
        if (maxClassScore > 0.1) { // 比较
            for(int x = 0; x < signalResultNum; ++x) {
                SPDLOG_DEBUG("========== {}", data[x]);
            }
            SPDLOG_DEBUG("----------------{}", maxClassScore);
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * resizeScales);
            int top = int((y - 0.5 * h) * resizeScales);

            int width = int(w * resizeScales);
            int height = int(h * resizeScales);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += signalResultNum;
    }
}

static void preProcess(cv::Mat& source, cv::Mat& target) {
    target = source.clone();
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
    cv::dnn::blobFromImage(target, input, 1.0F / 255, modelShape, cv::Scalar(), true, false);
    std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    postProcess(reinterpret_cast<float*>(input.data), inputNodeDims);
    // delete[] blob;
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
    cv::Mat frame;
    cv::Mat output;
    cv::VideoCapture camera(0);
    while(true) {
        camera >> frame;
        preProcess(frame, output);
        cv::imshow("face", frame);
        if(cv::waitKey(1) == 27) {
            break;
        }
    }
    releaseSession();
}
