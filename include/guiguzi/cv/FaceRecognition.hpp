#ifndef LFR_HEADER_GUIGUZI_FACE_RECOGNITION_HPP
#define LFR_HEADER_GUIGUZI_FACE_RECOGNITION_HPP

#include <string>

namespace ggguiguziz {

extern void onnx_face_recognition();

extern double opencv_face_recognition(const std::string& source, const std::string& target);
extern void   opencv_face_recognition(const std::string& model, const std::string& path, const std::string& face);

} // END OF guiguzi

#endif // END OF LFR_HEADER_GUIGUZI_FACE_RECOGNITION_HPP
