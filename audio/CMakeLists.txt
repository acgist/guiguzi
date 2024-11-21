add_subdirectory(rnnoise/rnnoise)

add_executable(
    testRnnoise
    rnnoise/Main.cpp
    rnnoise/guiguzi/Guiguzi.cpp
)
add_test(NAME testRnnoise COMMAND testRnnoise)
set_target_properties(testRnnoise PROPERTIES FOLDER "audio")
if(CMAKE_HOST_WIN32)
    target_link_libraries(
        testRnnoise
        rnnoise
        ${FFMPEG_LIBRARIES}
    )
else()
    target_link_libraries(
        testRnnoise
        rnnoise
        avutil
        avcodec
        avformat
    )
endif()
