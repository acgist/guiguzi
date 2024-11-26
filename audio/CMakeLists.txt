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
        swresample
    )
endif()

add_executable(
    testOpus
    rnnoise/Opus.cpp
    rnnoise/opus/Guiguzi.cpp
)
add_test(NAME testOpus COMMAND testOpus)
set_target_properties(testOpus PROPERTIES FOLDER "audio")
if(CMAKE_HOST_WIN32)
    target_link_libraries(
        testOpus
        rnnoise
        Opus::opus
    )
else()
    target_link_libraries(
        testOpus
        rnnoise
        opus
    )
endif()
