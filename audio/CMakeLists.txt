add_subdirectory(rnnoise/rnnoise)

add_executable(
    testRnnoise
    rnnoise/Main.cpp
)
add_test(NAME testRnnoise COMMAND testRnnoise)
set_target_properties(testRnnoise PROPERTIES FOLDER "audio")
target_link_libraries(
    testRnnoise
    rnnoise
)
