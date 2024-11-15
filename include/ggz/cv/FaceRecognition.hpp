#ifndef LFR_HEADER_GGZ_FACE_RECOGNITION_HPP
#define LFR_HEADER_GGZ_FACE_RECOGNITION_HPP

#include <string>

namespace ggz {

extern void yolo_face_recognition();

extern double opencv_face_recognition(const std::string& source, const std::string& target);
extern void   opencv_face_recognition(const std::string& model, const std::string& path, const std::string& face);

} // END OF ggz

#endif // END OF LFR_HEADER_GGZ_FACE_RECOGNITION_HPP
