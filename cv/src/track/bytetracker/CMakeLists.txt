project(bytetracker)

file(GLOB src_files src/*.cpp)

add_library(${PROJECT_NAME} ${src_files})

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${OpenCV_LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "module")
