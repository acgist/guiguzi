#include "Aliang.hpp"

#include <atomic>
#include <thread>

#include "spdlog/spdlog.h"

#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

#include "onnxruntime_cxx_api.h"

static Ort::Env*       env      { nullptr }; // ONNX运行环境
static std::atomic_int env_count{ 0       }; // ONNX环境计数

// 默认日志
static OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;

guiguzi::OnnxRuntime::OnnxRuntime(
    int wh,
    const char* logid,
    const std::vector<std::string>& classes,
    float confidenceThreshold,
    float iouThreshold
) : wh(wh),
    logid(logid),
    classes(classes),
    confidenceThreshold(confidenceThreshold),
    iouThreshold(iouThreshold)
{
    ++env_count;
    if(!env) {
        env = new Ort::Env(log_level, logid);
    }
    this->inputNodeDims.push_back(wh);
    this->inputNodeDims.push_back(wh);
}

guiguzi::OnnxRuntime::~OnnxRuntime() {
    if(env && --env_count == 0) {
        SPDLOG_DEBUG("释放ONNX运行环境：{}", this->logid);
        env->release();
        delete env;
        env = nullptr;
    }
    if(this->session) {
        SPDLOG_DEBUG("释放ONNX会话：{}", this->logid);
        this->session->release();
        delete this->session;
        this->session = nullptr;
    }
    if(this->runOptions) {
        SPDLOG_DEBUG("释放ONNX配置：{}", this->logid);
        this->runOptions->release();
        delete this->runOptions;
        this->runOptions = nullptr;
    }
    for(auto ptr : this->inputNodeNames) {
        delete[] ptr;
    }
    for(auto ptr : this->outputNodeNames) {
        delete[] ptr;
    }
}

bool guiguzi::OnnxRuntime::createSession(const std::string& path) {
    SPDLOG_DEBUG("创建会话：{} - {}", this->logid, path);
    Ort::SessionOptions options;
    options.SetLogSeverityLevel(static_cast<int>(log_level));
    options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    #if _WIN32
    std::wstring wPath(path.begin(), path.end());
    this->session = new Ort::Session(*env, wPath.c_str(), options);
    #else
    this->session = new Ort::Session(*env, path.c_str(), options);
    #endif
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t inputNodeCount  = this->session->GetInputCount();
    const size_t outputNodeCount = this->session->GetOutputCount();
    for(size_t index = 0; index < inputNodeCount; ++index) {
        Ort::AllocatedStringPtr name = this->session->GetInputNameAllocated(index, allocator);
        char* copy = new char[32];
        std::strcpy(copy, name.get());
        this->inputNodeNames.push_back(copy);
        SPDLOG_DEBUG("输入节点：{} - {}", index, copy);
    }
    for(size_t index = 0; index < outputNodeCount; ++ index) {
        Ort::AllocatedStringPtr name = this->session->GetOutputNameAllocated(index, allocator);
        char* copy = new char[32];
        std::strcpy(copy, name.get());
        this->outputNodeNames.push_back(copy);
        SPDLOG_DEBUG("输出节点：{} - {}", index, copy);
    }
    this->runOptions = new Ort::RunOptions(nullptr);
    return true;
}

Ort::Value guiguzi::OnnxRuntime::run(
    float* blob,               // 图片数据
    std::vector<int64_t>& dims // 结果维度
) {
    #ifdef __CUDA__
    // TODO: CUDA
    #else
    const Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        blob,
        3 * this->wh * this->wh,
        this->inputNodeDims.data(),
        this->inputNodeDims.size()
    );
    #endif
    auto outputTensor = this->session->Run(
        *this->runOptions,
        inputNodeNames.data(),
        &inputTensor,
        inputNodeNames.size(),
        outputNodeNames.data(),
        outputNodeNames.size()
    );
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    std::vector<int64_t> outputNodeDims = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
    dims.swap(outputNodeDims);
    return std::move(outputTensor.front());
}

void guiguzi::OnnxRuntime::run(
    float* blob,               // 图片数据
    std::vector<float>&   ret, // 结果数据
    std::vector<int64_t>& dims // 结果维度
) {
    auto output = this->run(blob, dims);
    int size = 1;
    for(const auto& dim : dims) {
        size *= dim;
    }
    ret.resize(size);
    float* data = output.GetTensorMutableData<float>();
    std::memcpy(ret.data(), data, ret.size() * sizeof(float));
}

void guiguzi::OnnxRuntime::run(
    float* blob,                   // 图片数据
    const float& scale,            // 图片缩放
    std::vector<cv::Rect> & boxes, // 框
    std::vector<cv::Point>& points // 关键点：眼睛、眼睛、鼻子、嘴巴、嘴巴
) {
    std::vector<int64_t> dims;
    auto output = this->run(blob, dims);
    const int64_t& signalResultNum = dims[1];
    const int64_t& strideNum       = dims[2];
    float* output_data = output.GetTensorMutableData<float>();
    cv::Mat rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output_data);
    rawData = rawData.t();
    float* data = reinterpret_cast<float*>(rawData.data);
    std::vector<int>   class_ids_ori;   // 类型
    std::vector<float> confidences_ori; // 置信度
    std::vector<cv::Rect>  boxes_ori;   // 框
    std::vector<cv::Point> points_ori;  // 关键点
    for (int index = 0; index < strideNum; ++index) {
        cv::Point class_id;   // 类别
        double maxConfidence; // 分数
        float* classesScores = data + 4;
        cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
        cv::minMaxLoc(scores, NULL, &maxConfidence, NULL, &class_id);
        if(maxConfidence > this->confidenceThreshold) {
            // 中心x 中心y 宽度 高度
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            int left   = int((x - 0.5 * w) * scale);
            int top    = int((y - 0.5 * h) * scale);
            int width  = int(w * scale);
            int height = int(h * scale);
            class_ids_ori.push_back(class_id.x);
            confidences_ori.push_back(maxConfidence);
            boxes_ori.push_back(cv::Rect(left, top, width, height));
            // 眼睛 眼睛 鼻子 嘴巴 嘴巴
            points_ori.push_back(cv::Point((data[5]  * scale), (data[6]  * scale)));
            points_ori.push_back(cv::Point((data[8]  * scale), (data[9]  * scale)));
            points_ori.push_back(cv::Point((data[11] * scale), (data[12] * scale)));
            points_ori.push_back(cv::Point((data[14] * scale), (data[15] * scale)));
            points_ori.push_back(cv::Point((data[17] * scale), (data[18] * scale)));
        }
        data += signalResultNum;
    }
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes_ori, confidences_ori, this->confidenceThreshold, this->iouThreshold, nmsResult);
    for(const auto& index : nmsResult) {
        boxes.push_back(boxes_ori[index]);
        points.push_back(points_ori[5 * index + 0]);
        points.push_back(points_ori[5 * index + 1]);
        points.push_back(points_ori[5 * index + 2]);
        points.push_back(points_ori[5 * index + 3]);
        points.push_back(points_ori[5 * index + 4]);
    }
}

void guiguzi::OnnxRuntime::run(
    float* blob,                     // 图片数据
    const float& scale,              // 图片缩放
    std::vector<int>  & class_ids,   // 类型
    std::vector<float>& confidences, // 置信度
    std::vector<cv::Rect>& boxes     // 框
) {
    std::vector<int64_t> dims;
    auto output = this->run(blob, dims);
    const int64_t& signalResultNum = dims[1];
    const int64_t& strideNum       = dims[2];
    float* output_data = output.GetTensorMutableData<float>();
    cv::Mat rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output_data);
    rawData = rawData.t();
    float* data = reinterpret_cast<float*>(rawData.data);
    std::vector<int> class_ids_ori;     // 类型
    std::vector<float> confidences_ori; // 置信度
    std::vector<cv::Rect> boxes_ori;    // 框
    for (int index = 0; index < strideNum; ++index) {
        cv::Point class_id;   // 类别
        double maxConfidence; // 分数
        float* classesScores = data + 4;
        cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
        cv::minMaxLoc(scores, NULL, &maxConfidence, NULL, &class_id);
        if(maxConfidence > this->confidenceThreshold) {
            // 中心x 中心y 宽度 高度
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            int left   = int((x - 0.5 * w) * scale);
            int top    = int((y - 0.5 * h) * scale);
            int width  = int(w * scale);
            int height = int(h * scale);
            class_ids_ori.push_back(class_id.x);
            confidences_ori.push_back(maxConfidence);
            boxes_ori.push_back(cv::Rect(left, top, width, height));
        }
        data += signalResultNum;
    }
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes_ori, confidences_ori, this->confidenceThreshold, this->iouThreshold, nmsResult);
    for(const auto& index : nmsResult) {
        class_ids.push_back(class_ids_ori[index]);
        confidences.push_back(confidences_ori[index]);
        boxes.push_back(boxes_ori[index]);
    }
}
