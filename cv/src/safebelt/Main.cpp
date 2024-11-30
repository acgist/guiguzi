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
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Safebelt");
    Ort::SessionOptions options;
    options.SetLogSeverityLevel(3);
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    #if _WIN32
    session = new Ort::Session(*env, L"D:/tmp/safebelt/best.onnx", options);
    #else 
    session = new Ort::Session(*env, "D:/tmp/safebelt/safebelt.onnx", options);
    #endif
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

    // int col = source.cols;
    // int row = source.rows;
    // int _max = MAX(col, row);
    // cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    // result.setTo(cv::Scalar(255,255,255));
    // cv::imshow("x", result);
    // cv::waitKey(0);
    // source.copyTo(result(cv::Rect(0, 0, col, row)));
    // cv::imshow("x", result);
    // cv::waitKey(0);
    // resizeScales = 1.0F * _max / 640;

    // cv::Mat result = source.clone();
    // if (source.cols >= source.rows)
    // {
    //     resizeScales = source.cols / (float)modelShape.width;
    //     cv::resize(result, result, cv::Size(modelShape.width, int(source.rows / resizeScales)));
    // }
    // else
    // {
    //     resizeScales = source.rows / (float)modelShape.width;
    //     cv::resize(result, result, cv::Size(int(source.cols / resizeScales), modelShape.height));
    // }
    // cv::Mat tempImg = cv::Mat::zeros(modelShape.width, modelShape.height, CV_8UC3);
    // source.copyTo(tempImg(cv::Rect(0, 0, source.cols, source.rows)));
    // result = tempImg;

    return result;

// static cv::Mat formatToSquare(const cv::Mat &iImg) {
//     cv::Mat oImg = iImg.clone();
//     if (iImg.cols >= iImg.rows)
//     {
//         resizeScales = iImg.cols / (float)modelShape.width;
//         cv::resize(oImg, oImg, cv::Size(modelShape.width, int(iImg.rows / resizeScales)));
//     }
//     else
//     {
//         resizeScales = iImg.rows / (float)modelShape.width;
//         cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), modelShape.height));
//     }
//     cv::Mat tempImg = cv::Mat::zeros(modelShape.width, modelShape.height, CV_8UC3);
//     oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
//     oImg = tempImg;
//     cv::imshow("v", oImg);
//     cv::waitKey(0);

    // return oImg;
}

static std::vector<cv::Rect> postProcess(float* blob, std::vector<int64_t>& inputNodeDims, std::vector<float>& percent, std::vector<int64_t>& classify) {
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
    // 转换
    int signalResultNum = outputNodeDims[1];
    int strideNum       = outputNodeDims[2];
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat rawData;
    rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
    rawData = rawData.t();
    float* data = (float*) rawData.data;
    float zxd = 0.4;
    for (int i = 0; i < strideNum; ++i) {
        // if(data[4] > 0.1 || data[5] > 0.1 || data[6] > 0.1 || data[7] > 0.1) {
        //     SPDLOG_DEBUG("----------------{:.6f} {:.6f} {:.6f} {:.6f}", data[4], data[5], data[6], data[7]);
        // }
        float* classesScores = data + 4;
        cv::Point class_id;
        cv::Mat scores(1, 3, CV_32FC1, classesScores);
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if (maxClassScore > zxd) { // 置信度
            // for(int x = 0; x < signalResultNum; ++x) {
            //     SPDLOG_DEBUG("========== {}", data[x]);
            // }
            // SPDLOG_DEBUG("----------------{} {}", class_id.x, maxClassScore);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * resizeScales);
            int top  = int((y - 0.5 * h) * resizeScales);

            int width  = int(w * resizeScales);
            int height = int(h * resizeScales);

            boxes.push_back(cv::Rect(left, top, width, height));
            class_ids.push_back(class_id.x);
            confidences.push_back(maxClassScore);
        }
        data += signalResultNum;
    }
    std::vector<int> nmsResult;
    std::vector<cv::Rect> rest;
    cv::dnn::NMSBoxes(boxes, confidences, zxd, 0.8, nmsResult);
    for(const auto& i : nmsResult) {
        rest.push_back(boxes[i]);
        percent.push_back(confidences[i]);
        classify.push_back(class_ids[i]);
        SPDLOG_DEBUG("类型：{}", class_ids[i]);
    }
    return rest;
}

template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return 0;
}

char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    if (iImg.channels() == 3)
    {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }

        if (iImg.cols >= iImg.rows)
        {
            resizeScales = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
        }
        else
        {
            resizeScales = iImg.rows / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
        }
        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg;
    return 0;
}

[[maybe_unused]] static void run(cv::Mat& source) {
    cv::Mat input;
    cv::Mat target = source.clone();

    // cv::Mat processedImg;
    // std::vector<int> imgSize = { 640, 640 };
    // PreProcess(target, imgSize, processedImg);
    // float* blob = new float[processedImg.total() * 3];
    // BlobFromImage(processedImg, blob);

    target = formatToSquare(target);
    cv::dnn::blobFromImage(target, input, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    float* blob = reinterpret_cast<float*>(input.data);
    
    std::vector<int64_t> inputNodeDims = { 1, 3, 640, 640 };
    std::vector<float> percent;
    std::vector<int64_t> classify;
    // postProcess(reinterpret_cast<float*>(input.data), inputNodeDims);
    std::vector<cv::Rect> boxs = postProcess(blob, inputNodeDims, percent, classify);
    auto label = percent.begin();
    auto classify_name = classify.begin();
    const char* classify_names[] { /* "badge", */ "ground", "safebelt", "offground" };
    const cv::Scalar color_classes[] {
        /* cv::Scalar(255, 255, 255), */
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 255),
        cv::Scalar(255, 255, 0)
    };
    for(const auto& rect : boxs) {
        cv::rectangle(source, rect, cv::Scalar{ 255, 255, 255 });
        cv::putText(
            source,
            std::to_string(*label),
            cv::Point(rect.x, rect.y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            color_classes[*classify_name],
            1
        );
        cv::putText(
            source,
            classify_names[*classify_name],
            cv::Point(rect.x, rect.y + 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            color_classes[*classify_name],
            1
        );
        ++label;
        ++classify_name;
    }
}

static void runImage(const char* path) {
    auto input = cv::imread(path);
    float scale = 800.0F / MIN(input.cols, input.rows);
    // 图片缩小
    cv::resize(input, input, cv::Size(), scale, scale);
    run(input);
    cv::namedWindow("input");
    cv::imshow("input", input);
    cv::waitKey(0);
}

int main() {
    guiguzi::logger::init();
    createSession();
    runImage("D:/tmp/safebelt/0.jpg");
    runImage("D:/tmp/safebelt/1.jpg");
    runImage("D:/tmp/safebelt/2.jpg");
    runImage("D:/tmp/safebelt/3.jpg");
    runImage("D:/tmp/safebelt/4.jpg");
    runImage("D:/tmp/safebelt/5.jpg");
    runImage("D:/tmp/safebelt/6.jpg");
    runImage("D:/tmp/safebelt/7.jpg");
    runImage("D:/tmp/safebelt/8.jpg");
    runImage("D:/tmp/safebelt/9.jpg");
    runImage("D:/tmp/safebelt/10.jpg");
    runImage("D:/tmp/safebelt/11.jpg");
    runImage("D:/tmp/safebelt/12.jpg");
    runImage("D:/tmp/safebelt/13.jpg");
    runImage("D:/tmp/safebelt/14.jpg");
    // cv::VideoCapture capture(0);
    // capture.isOpened();
    // cv::Mat frame;
    // // cv::VideoWriter writer(
    // //     "D:/tmp/camera.mp4",
    // //     cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
    // //     25,
    // //     cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT))
    // // );
    // while(true) {
    //     capture >> frame;
    //     run(frame);
    //     // writer.write(frame);
    //     cv::imshow("input", frame);
    //     auto key = cv::waitKey(10);
    //     if(key == 'q') {
    //         break;
    //     }
    // }
    guiguzi::logger::shutdown();
    return 0;
}