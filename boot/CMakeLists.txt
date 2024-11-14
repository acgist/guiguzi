project(ggzboot VERSION 1.0.0 LANGUAGES C CXX)

add_library(
    ${PROJECT_NAME}
    src/Logger.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")

if(CMAKE_HOST_WIN32)
    target_link_libraries(
        ${PROJECT_NAME}
        spdlog::spdlog
        ${OpenCV_LIBS}
    )
else()
    target_link_libraries(
        ${PROJECT_NAME}
        fmt
        spdlog
        ${OpenCV_LIBS}
    )
endif()

add_executable(ggz src/Main.cpp)

if(CMAKE_HOST_WIN32)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRT")
    endif()

    add_custom_command(
        TARGET ggz POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DLLS} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPDLOG_DLLS} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
endif()
