add_executable(testFaceDetection src/face-detection/Main.cpp src/face-detection/FDOpenCV.cpp)
add_test(NAME testFaceDetection COMMAND testFaceDetection)
set_target_properties(testFaceDetection PROPERTIES FOLDER "cv")
target_link_libraries(
    testFaceDetection
    ggzboot
)

add_executable(testFaceRecognition src/face-recognition/Main.cpp src/face-recognition/FROpenCV.cpp)
add_test(NAME testFaceRecognition COMMAND testFaceRecognition)
set_target_properties(testFaceRecognition PROPERTIES FOLDER "cv")
target_link_libraries(
    testFaceRecognition
    ggzboot
)
