add_executable(
    testVideoCapture
    src/video-capture/Main.cpp
)
add_test(NAME testVideoCapture COMMAND testVideoCapture)
set_target_properties(testVideoCapture PROPERTIES FOLDER "cv")
target_link_libraries(
    testVideoCapture
    ggzboot
)

add_executable(
    testFaceDetection
    src/face-detection/Main.cpp
    src/face-detection/FDOnnx.cpp
    src/face-detection/FDOpenCV.cpp
    src/face-detection/FDLibTorch.cpp
)
add_test(NAME testFaceDetection COMMAND testFaceDetection)
set_target_properties(testFaceDetection PROPERTIES FOLDER "cv")
target_link_libraries(
    testFaceDetection
    ggzboot
)

add_executable(
    testFaceRecognition
    src/face-recognition/Main.cpp
    src/face-recognition/FROpenCV.cpp
)
add_test(NAME testFaceRecognition COMMAND testFaceRecognition)
set_target_properties(testFaceRecognition PROPERTIES FOLDER "cv")
target_link_libraries(
    testFaceRecognition
    ggzboot
)
