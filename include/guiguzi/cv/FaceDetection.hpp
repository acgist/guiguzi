#ifndef LFR_HEADER_GUIGUZI_FACE_DETECTION_HPP
#define LFR_HEADER_GUIGUZI_FACE_DETECTION_HPP

#include <string>

namespace guiguzi {

extern void onnx_face_detection();

extern void opencv_face_detection(const std::string& model, const std::string& path);

extern void libtorch_face_detection(const std::string& model, const std::string& path);

} // END OF guiguzi

#endif // END OF LFR_HEADER_GUIGUZI_FACE_DETECTION_HPP
