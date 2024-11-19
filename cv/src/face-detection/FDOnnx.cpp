/**
 * https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
 * https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-1.19.2.zip
 * 
 * https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime-CPP/inference.cpp
 */

#include "guiguzi/cv/FaceDetection.hpp"

#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

#include "onnxruntime_cxx_api.h"

#define min(a,b) (((a) < (b)) ? (a) : (b))

enum class MODEL_TYPE
{
    //FLOAT32 MODEL
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,

    //FLOAT16 MODEL
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
    YOLO_CLS_HALF = 6
};

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;

static Ort::Env* env;
static Ort::Session* session;
static Ort::RunOptions options { nullptr };

static std::vector<std::string> classes{};

static float rectConfidenceThreshold = 0.2F;
static float iouThreshold;
static float resizeScales;

static MODEL_TYPE modelType = MODEL_TYPE::YOLO_DETECT_V8;

static std::vector<int> imgSize{ 640, 640 };

static cv::Size2f modelShape = cv::Size(640, 640);

static std::vector<const char*> inputNodeNames;
static std::vector<const char*> outputNodeNames;

static void release() {
    delete session;
    delete env;
}

template<typename T>
static void BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type((iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
}

static void PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
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

    switch (modelType)
    {
    case MODEL_TYPE::YOLO_DETECT_V8:
    case MODEL_TYPE::YOLO_POSE:
    case MODEL_TYPE::YOLO_DETECT_V8_HALF:
    case MODEL_TYPE::YOLO_POSE_V8_HALF://LetterBox
    {
        // cv::imshow("a1", iImg.clone());
        if (iImg.cols >= iImg.rows)
        {
            resizeScales = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
        }
        else
        {
            resizeScales = iImg.rows / (float)iImgSize.at(1);
            cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
        }
        // cv::imshow("a2", iImg.clone());
        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg;
        // cv::imshow("a3", iImg.clone());
        break;
    }
    case MODEL_TYPE::YOLO_CLS://CenterCrop
    {
        int h = iImg.rows;
        int w = iImg.cols;
        int m = min(h, w);
        int top = (h - m) / 2;
        int left = (w - m) / 2;
        cv::resize(oImg(cv::Rect(left, top, m, m)), oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
        break;
    }
    }
}

static void createSession() {
    std::string lines = R"(0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush)";
    std::stringstream ss(lines);
    std::string line;
    while(std::getline(ss, line)) {
        classes.push_back(line);
    }
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
    Ort::SessionOptions sessionOption;
    // OrtCUDAProviderOptions cudaOption;
    // cudaOption.device_id = 0;
    // sessionOption.AppendExecutionProvider_CUDA(cudaOption);
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOption.SetIntraOpNumThreads(1);
    sessionOption.SetLogSeverityLevel(3);
    const wchar_t* modelPath = L"D:/tmp/face/yolov5su.onnx";
    // const wchar_t* modelPath = L"D:/tmp/face/yolo11n.onnx";
    session = new Ort::Session(*env, modelPath, sessionOption);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++)
    {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
    }
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++)
    {
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
    }
    options = Ort::RunOptions{ nullptr };
}

template<typename N>
static void TensorProcess(cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
    std::vector<DL_RESULT>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    switch (modelType)
    {
    case MODEL_TYPE::YOLO_DETECT_V8:
    case MODEL_TYPE::YOLO_DETECT_V8_HALF:
    {
        int signalResultNum = outputNodeDims[1];//84
        int strideNum = outputNodeDims[2];//8400
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        cv::Mat rawData;
        if (modelType == MODEL_TYPE::YOLO_DETECT_V8)
        {
            // FP32
            rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
        }
        else
        {
            // FP16
            rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        // Note:
        // ultralytics add transpose operator to the output of yolov8 model.which make yolov8/v5/v7 has same shape
        // https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
        rawData = rawData.t();

        float* data = (float*)rawData.data;

        for (int i = 0; i < strideNum; ++i)
        {
            float* classesScores = data + 4;
            cv::Mat scores(1, classes.size(), CV_32FC1, classesScores);
            cv::Point class_id;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > rectConfidenceThreshold)
            {
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
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
        for (int i = 0; i < nmsResult.size(); ++i)
        {
            int idx = nmsResult[i];
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            oResult.push_back(result);
        }

        break;
    }
    case MODEL_TYPE::YOLO_CLS:
    case MODEL_TYPE::YOLO_CLS_HALF:
    {
        cv::Mat rawData;
        if (modelType == MODEL_TYPE::YOLO_CLS) {
            // FP32
            rawData = cv::Mat(1, classes.size(), CV_32F, output);
        } else {
            // FP16
            rawData = cv::Mat(1, classes.size(), CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        float *data = (float *) rawData.data;

        DL_RESULT result;
        for (int i = 0; i < classes.size(); i++)
        {
            result.classId = i;
            result.confidence = data[i];
            oResult.push_back(result);
        }
        break;
    }
    default:
        std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
    }
}

static cv::Mat formatToSquare(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

static void RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
    // cv::Mat processedImg;
    // PreProcess(iImg, imgSize, processedImg);
    cv::Mat processedImg = iImg.clone();
    resizeScales = 1.0F * std::max(iImg.cols, iImg.rows) / 640;
    if (static_cast<int>(modelType) < 4)
    {
        // float* blob = new float[processedImg.total() * 3];
        // BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        if (modelShape.width == modelShape.height) {
            processedImg = formatToSquare(processedImg);
        }
        cv::Mat ret;
        cv::dnn::blobFromImage(processedImg, ret, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
        float* blob = reinterpret_cast<float*>(ret.data);
        TensorProcess(iImg, blob, inputNodeDims, oResult);
        // delete[] blob;
    }
}

void guiguzi::onnx_face_detection() {
    createSession();
    std::string img_path = "D:/tmp/F4.jpg";
    cv::Mat img = cv::imread(img_path);
    std::vector<DL_RESULT> res;
    RunSession(img, res);

    for (auto& re : res)
    {
        cv::RNG rng(cv::getTickCount());
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        cv::rectangle(img, re.box, color, 3);

        float confidence = floor(100 * re.confidence) / 100;
        std::cout << std::fixed << std::setprecision(2);
        std::string label = classes[re.classId] + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        cv::rectangle(
            img,
            cv::Point(re.box.x, re.box.y - 25),
            cv::Point(re.box.x + label.length() * 15, re.box.y),
            color,
            cv::FILLED
        );

        cv::putText(
            img,
            label,
            cv::Point(re.box.x, re.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 0, 0),
            2
        );


    }
    std::cout << "Press any key to exit" << std::endl;
    cv::imshow("Result of Detection", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
