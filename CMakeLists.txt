cmake_minimum_required(VERSION 3.22.0)

#[[
    鬼谷子
    https://gitee.com/acgist/guiguzi
    https://github.com/acgist/guiguzi
]]

# https://cmake.org/cmake/help/latest/

project(guiguzi-project VERSION 1.0.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(BUILD_SHARED_LIBS "是否编译为共享库" OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
if(CMAKE_HOST_UNIX)
    set(CMAKE_C_STANDARD 23)
    set(CMAKE_CXX_STANDARD 23)
    # https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c2x -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers -O0 -g")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers -O0 -g")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wunused-variable -Wunused-parameter -Wmissing-field-initializers -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wunused-variable -Wunused-parameter -Wmissing-field-initializers -O3")
    if(BUILD_SHARED_LIBS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
elseif(CMAKE_HOST_WIN32)
    if(MSVC)
        # TODO: 使用C23/C++23
        set(CMAKE_C_STANDARD 17)
        set(CMAKE_CXX_STANDARD 20)
        # https://learn.microsoft.com/zh-cn/cpp/build/reference/compiler-options-listed-by-category
        # /wd4804 = 忽略告警
        # /wd4819 = 忽略告警
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /std:c17 /std:c++20")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c17 /std:c++20")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /wd4804 /wd4819")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /wd4804 /wd4819")
        # set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /Ob2")
        # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2")
        add_compile_options("/utf-8")
        add_compile_options("/Zc:__cplusplus")
    else()
        message(WARNING "需要安装MSVC")
    endif()
else()
    message(FATAL_ERROR "不支持的系统：${CMAKE_HOST_SYSTEM_NAME}")
endif()

message(STATUS "当前模式: ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin)

set(
    CMAKE_PREFIX_PATH
    ${CMAKE_PREFIX_PATH}
    deps/ffmpeg
    deps/libtorch
)

include_directories(include deps/ffmpeg/include deps/onnxruntime/include)

if(CMAKE_HOST_UNIX)
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        deps/onnxruntime
    )

    find_package(onnxruntime REQUIRED)

    link_directories(deps/ffmpeg/lib)

    # add_library(
    #     avformat
    #     SHARED
    #     IMPORTED
    # )
    # set_target_properties(
    #     avformat
    #     PROPERTIES IMPORTED_LOCATION
    #     deps/ffmpeg/lib/libavformat.so
    # )
endif()

if(CMAKE_HOST_WIN32)
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        deps/opus
        deps/eigen3
        deps/opencv
        deps/spdlog
    )

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        file(GLOB OPUS_DLLS   "deps/opus/debug/bin/*.dll")
        file(GLOB FFMPEG_DLLS "deps/ffmpeg/debug/bin/*.dll")
        file(GLOB SPDLOG_DLLS "deps/spdlog/debug/bin/*.dll")
        file(GLOB OPENCV_DLLS "deps/opencv/x64/vc16/bin/opencv_world4100d.dll" "deps/opencv/x64/vc16/bin/opencv_videoio_ffmpeg4100_64.dll")
    else()
        file(GLOB OPUS_DLLS   "deps/opus/bin/*.dll")
        file(GLOB FFMPEG_DLLS "deps/ffmpeg/bin/*.dll")
        file(GLOB SPDLOG_DLLS "deps/spdlog/bin/*.dll")
        file(GLOB OPENCV_DLLS "deps/opencv/x64/vc16/bin/opencv_world4100.dll" "deps/opencv/x64/vc16/bin/opencv_videoio_ffmpeg4100_64.dll")
    endif()
    file(GLOB TORCH_DLLS       "deps/libtorch/lib/*.dll")
    file(GLOB ONNXRUNTIME_DLLS "deps/onnxruntime/lib/*.dll")

    file(GLOB ONNXRUNTIME_LIBS "deps/onnxruntime/lib/onnxruntime.lib")

    find_package(Opus   REQUIRED)
    find_package(FFMPEG REQUIRED)
endif()

find_package(Torch  REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(cv)
add_subdirectory(nlp)
add_subdirectory(boot)
add_subdirectory(deps)
add_subdirectory(audio)
