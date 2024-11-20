add_executable(
    testRnnoise
    rnnoise/rnnoise/rnn.c
    rnnoise/rnnoise/nnet.c
    rnnoise/rnnoise/pitch.c
    rnnoise/rnnoise/denoise.c
    rnnoise/rnnoise/kiss_fft.c
    rnnoise/rnnoise/celt_lpc.c
    rnnoise/rnnoise/nnet_default.c
    rnnoise/rnnoise/rnnoise_data.c
    rnnoise/rnnoise/rnnoise_tables.c
    rnnoise/rnnoise/parse_lpcnet_weights.c
)
add_test(NAME testRnnoise COMMAND testRnnoise)
set_target_properties(testRnnoise PROPERTIES FOLDER "audio")