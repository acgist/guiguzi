add_executable(bytetracker-onnx onnx/main.cpp)
set_target_properties(bytetracker-onnx PROPERTIES FOLDER "cv")
target_link_libraries(bytetracker-onnx bytetracker ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})

add_executable(bytetracker-opencv opencv/main.cpp)
set_target_properties(bytetracker-opencv PROPERTIES FOLDER "cv")
target_link_libraries(bytetracker-opencv bytetracker ${OpenCV_LIBS})
