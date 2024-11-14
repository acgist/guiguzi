add_executable(testFaceDetection Main.cpp Yolo.cpp OpenCV.cpp)
add_test(NAME testFaceDetection COMMAND testFaceDetection)
set_target_properties(testFaceDetection PROPERTIES FOLDER "test-face-detection")
target_link_libraries(
    testFaceDetection
    ggzboot
)
