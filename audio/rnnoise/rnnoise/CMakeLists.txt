cmake_minimum_required(VERSION 3.22.0)

project(rnnoise VERSION 1.0.0 LANGUAGES C)

include_directories(../guiguzi)

add_library(
    ${PROJECT_NAME}
    src/rnn.c
    src/nnet.c
    src/pitch.c
    src/denoise.c
    src/kiss_fft.c
    src/celt_lpc.c
    src/nnet_default.c
    src/rnnoise_data.c
    src/rnnoise_tables.c
    src/parse_lpcnet_weights.c
)

set_source_files_properties(
    ${SOURCE_FILES} PROPERTIES COMPILE_FLAGS -Wall -Wextra -Wpedantic
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "audio")
