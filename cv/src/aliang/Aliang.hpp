/**
 * 人脸识别 + 物体检测
 */

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>

#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"

namespace Ort {

    struct Env;
    struct Value;
    struct Session;
    struct RunOptions;

} // END OF Ort

namespace guiguzi {

// 处理图片
extern float* formatImage(int wh, const cv::Mat& source, cv::Mat& target, float& scale);

/**
 * ONNX运行环境
 */
class OnnxRuntime {

public:
    Ort::Env       * env       { nullptr }; // ONNX环境
    Ort::Session   * session   { nullptr }; // ONNX会话
    Ort::RunOptions* runOptions{ nullptr }; // ONNX运行配置
    std::vector<const char*> inputNodeNames;  // ONNX输入参数
    std::vector<const char*> outputNodeNames; // ONNX输出参数
    std::vector<int64_t>     inputNodeDims { 1, 3 }; // 输入参数维度
    int wh;
    std::vector<std::string> classes; // 分类
    float confidenceThreshold = 0.4F; // 置信度阈值
    float iouThreshold        = 0.6F; // IOU阈值

public:
    OnnxRuntime(int wh);
    OnnxRuntime(int wh, const std::vector<std::string>& classes, float confidenceThreshold = 0.4F, float iouThreshold = 0.6F);
    virtual ~OnnxRuntime();

public:
    // 创建会话
    bool createSession(const std::string& path, const char* logid);
    // 执行计算
    Ort::Value run(float* blob, int64_t& signalResultNum, int64_t& strideNum);
    void       run(float* blob, std::vector<float>& ret, int64_t& signalResultNum, int64_t& strideNum);
    void       run(
        float* blob,                   // 图片数据
        const float& scale,            // 图片缩放
        std::vector<cv::Rect>& boxes,  // 框
        std::vector<cv::Point>& points // 关键点：眼睛、鼻子、嘴巴
    );
    void       run(
        float* blob,                     // 图片数据
        const float& scale,              // 图片缩放
        std::vector<int>& class_ids,     // 类型
        std::vector<float>& confidences, // 置信度
        std::vector<cv::Rect>& boxes     // 框
    );

};

class Recognition;

/**
 * 物体检测
 * 人脸、人体、安全帽、安全带
 */
class Detection {

public:
    std::vector<std::string> classes; // 分类
    std::unique_ptr<OnnxRuntime> model{ nullptr }; // 检测模型

public:
    Detection(const std::string& model, const char* logid, const std::vector<std::string>& classes, float confidenceThreshold = 0.4F, float iouThreshold = 0.6F);

public:
    // 物体识别
    void detection(
        const cv::Mat& image,            // 图片
        std::vector<int>& class_ids,     // 类型
        std::vector<float>& confidences, // 置信度
        std::vector<cv::Rect>& boxes     // 框
    );
    // 物体识别标记
    void detection(cv::Mat& image, Recognition& recognition);

};

/**
 * 人脸识别
 */
class Recognition {

public:
    float threshold = 0.8F; // 阈值
    std::unique_ptr<OnnxRuntime> faceModel   { nullptr }; // 人脸模型
    std::unique_ptr<OnnxRuntime> featureModel{ nullptr }; // 特征模型
    std::map<std::string, std::vector<std::vector<float>>> features; // 特征

public:
    Recognition(
        const std::string& face_model,    const char* face_logid,
        const std::string& feature_model, const char* feature_logid,
        float threshold = 0.8F, float confidenceThreshold = 0.4F, float iouThreshold = 0.6F
    );

public:
    // 人脸提取
    bool extract(cv::Mat& image, std::vector<cv::Rect>& boxes, std::vector<cv::Point>& points);
    // 人脸居中
    bool center(cv::Mat& image, std::vector<cv::Point>& points);
    // 特征提取
    void feature(cv::Mat& image, std::vector<float>& feature);
    // 特征比较
    double compare(const std::vector<float>& source, const std::vector<float>& target);
    // 输入人脸
    void storage(const std::string& name, std::vector<cv::Mat>& images);
    void storage(const std::string& name, const std::vector<std::string>& images);
    // 人脸识别
    std::pair<std::string, double> recognition(cv::Mat& image);

};

} // END OF guiguzi
