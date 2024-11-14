#ifndef LFR_HEADER_GGZ_FACE_DETECTION_HPP
#define LFR_HEADER_GGZ_FACE_DETECTION_HPP

#include <string>

namespace ggz {

extern void dlib_face_detection();

extern void yolo_face_detection();

extern void opencv_face_detection(const std::string& model, const std::string& path);

} // END OF ggz

#endif // END OF LFR_HEADER_GGZ_FACE_DETECTION_HPP
