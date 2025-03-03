project(bytetrack-onnx)

file(GLOB src_files src/*.cpp)

add_executable(${PROJECT_NAME} ${src_files})

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})
