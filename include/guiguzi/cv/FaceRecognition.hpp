#ifndef LFR_HEADER_GUIGUZI_FACE_RECOGNITION_HPP
#define LFR_HEADER_GUIGUZI_FACE_RECOGNITION_HPP

#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

namespace guiguzi {

extern std::vector<float> onnx_face_feature(const cv::Mat& face);

extern void onnx_face_feature_center(const cv::Mat& face, std::vector<cv::Point>& s);

extern double onnx_face_feature_diff(const std::vector<float>& source, const std::vector<float>& target);

extern void onnx_face_recognition();

extern double opencv_face_recognition(const std::string& source, const std::string& target);
extern void   opencv_face_recognition(const std::string& model, const std::string& path, const std::string& face);

} // END OF guiguzi

#endif // END OF LFR_HEADER_GUIGUZI_FACE_RECOGNITION_HPP
