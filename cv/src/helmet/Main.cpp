#include "guiguzi/Logger.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/dnn/dnn.hpp"

#include "spdlog/spdlog.h"

#include "onnxruntime_cxx_api.h"

static Ort::Env* env { nullptr };
static Ort::Session* session { nullptr };
static Ort::RunOptions options { nullptr };

static std::vector<const char*> inputNodeNames;
static std::vector<const char*> outputNodeNames;

[[maybe_unused]] static void createSession() {
    if(env) {
        delete env;
        env = nullptr;
    }
    if(session) {
        delete session;
        session = nullptr;
    }
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Helmet");
    Ort::SessionOptions options;
    options.SetLogSeverityLevel(3);
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = new Ort::Session(*env, L"D:/tmp/helmet/helment.onnx", options);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++) {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
    }
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++) {
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
    }
    ::options = Ort::RunOptions{ nullptr };
}

static float resizeScales;
static cv::Size2f modelShape = cv::Size(640, 640);

static cv::Mat formatToSquare(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    resizeScales = 1.0F * _max / 640;
    return result;
}

static std::vector<cv::Rect> postProcess(float* blob, std::vector<int64_t>& inputNodeDims) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<float*>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        blob,
        3 * 640 * 640,
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
    for (int i = 0; i < strideNum; ++i) {
        float* classesScores = data + 4;
        // 20 - 4
        cv::Point class_id;
        cv::Mat scores(1, 2, CV_32FC1, classesScores);
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        // for(int x = 0; x < signalResultNum; ++x) {
        //     SPDLOG_DEBUG("========== {}", data[x]);
        // }
        // SPDLOG_DEBUG("----------------{}", maxClassScore);
        if (maxClassScore > 0.1) { // 置信度
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
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += signalResultNum;
    }
    std::vector<int> nmsResult;
    std::vector<cv::Rect> rest;
    cv::dnn::NMSBoxes(boxes, confidences, 0.1, 0.0, nmsResult);
    for(const auto& i : nmsResult) {
        rest.push_back(boxes[i]);
        SPDLOG_DEBUG("类型：{}", class_ids[i]);
    }
    return rest;
}

[[maybe_unused]] static void run(cv::Mat& source) {
    cv::Mat input;
    cv::Mat target = source.clone();
    target = formatToSquare(target);
    cv::dnn::blobFromImage(target, input, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    float* blob = reinterpret_cast<float*>(input.data);
    std::vector<int64_t> inputNodeDims = { 1, 3, 640, 640 };
    // postProcess(reinterpret_cast<float*>(input.data), inputNodeDims);
    std::vector<cv::Rect> boxs = postProcess(blob, inputNodeDims);
    for(const auto& rect : boxs) {
        cv::rectangle(source, rect, cv::Scalar{ 255, 0, 0 });
    }
}

int main() {
    guiguzi::logger::init();
    createSession();
    auto input = cv::imread("D:/tmp/helmet/train/1.jpg");
    // auto input = cv::imread("D:/tmp/helmet/val/34.jpg");
    run(input);
    cv::namedWindow("input", cv::WINDOW_NORMAL);
    cv::imshow("input", input);
    // cv::VideoCapture capture(0);
    // capture.isOpened();
    // cv::Mat frame;
    // while(true) {
    //     capture >> frame;
    //     run(frame);
    //     cv::imshow("input", frame);
    //     cv::waitKey(10);
    // }
    cv::waitKey(0);
    guiguzi::logger::shutdown();
    return 0;
}