#include "guiguzi/cv/FaceRecognition.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "onnxruntime_cxx_api.h"

static Ort::Env* env;
static Ort::Session* session;
static Ort::RunOptions options{ nullptr };

static std::vector<const char*> inputNodeNames;
static std::vector<const char*> outputNodeNames;

static void createSession() {
    if(env) {
        delete env;
        env = nullptr;
    }
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Face_Feature");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetLogSeverityLevel(3);
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if(session) {
        delete session;
        session = nullptr;
    }
    session = new Ort::Session(*env, L"D:/tmp/face/arcface_w600k_r50.onnx", sessionOptions);
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

static std::vector<float> postProcess(cv::Mat& source) {
    cv::Mat target = source.clone();
    cv::Mat input;
    cv::dnn::blobFromImage(target, input, 1.0F / 255, cv::Size{ 112, 112 }, cv::Scalar(), true, false);
    std::vector<int64_t> inputNodeDims = { 1, 3, 112, 112 };
    // postProcess(reinterpret_cast<float*>(input.data), inputNodeDims);
    // float* blob = new float[3 * 112 * 112];
    float* blob = reinterpret_cast<float*>(input.data);
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<float*>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        blob,
        3 * 112 * 112,
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
    // int signalResultNum = outputNodeDims[1]; // 20
    // int strideNum       = outputNodeDims[2]; // 8400
    // delete[] blob;
    std::vector<float> ret;
    ret.resize(512);
    std::copy_n(output, 512, ret.data());
    return ret;
}

double guiguzi::onnx_face_feature_diff(const std::vector<float>& source, const std::vector<float>& target) {
    double v = 0.0;
    cv::normalize(source, source);
    cv::normalize(target, target);
    for(int i = 0; i < source.size(); ++i) {
        v += std::pow((source[i] - target[i]), 2);
    }
    return v;
}

void guiguzi::onnx_face_feature_center(const cv::Mat& source, std::vector<cv::Point>& s) {
    std::vector<cv::Point> t;

    // t.push_back(cv::Point(114.0, 194.0));
    // t.push_back(cv::Point(221.0, 194.0));
    // t.push_back(cv::Point(165.0, 269.0));
    // t.push_back(cv::Point(135.0, 312.0));
    // t.push_back(cv::Point(227.0, 312.0));

    t.push_back(cv::Point(38.2946, 51.6963));
    t.push_back(cv::Point(73.5318, 51.5014));
    t.push_back(cv::Point(56.0252, 71.7366));
    t.push_back(cv::Point(41.5493, 92.3655));
    t.push_back(cv::Point(70.7299, 92.2041));

    auto M = cv::estimateAffinePartial2D(s, t);
    cv::imshow("bs1", source.clone());
    cv::warpAffine(source, source, M, source.size());
    // cv::warpAffine(source, source, M, cv::Size(112, 112), 3.0);
    cv::imshow("bs2", source.clone());
    cv::waitKey(0);
}

std::vector<float> guiguzi::onnx_face_feature(const cv::Mat& face) {
    if(!session) {
        createSession();
    }
    cv::imshow("face", face);
    auto min = std::min(face.rows, face.cols);
    cv::Rect rect((face.cols - min) / 2, (face.rows - min) / 2, min, min);
    cv::Mat input = face(rect);
    // cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    cv::resize(input, input, cv::Size(112, 112));
    cv::imshow("input", input);
    return postProcess(input);
}
