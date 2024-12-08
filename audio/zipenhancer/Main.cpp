/**
 * https://github.com/modelscope/modelscope/blob/master/modelscope/models/audio/ans/zipenhancer.py
 */

#include <vector>
#include <fstream>
#include <iostream>

#include "torch/torch.h"
#include "torch/script.h"

static int n_fft    = 400;
static int hop_size = 100;
static int win_size = 400;

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mag_pha_stft(
    torch::Tensor y,
    float compress_factor = 0.3
) {
    auto window = torch::hann_window(win_size);
    auto stft_spec = torch::stft(y, n_fft, hop_size, win_size, window, true, "reflect", false, std::nullopt, true);
    stft_spec = torch::view_as_real(stft_spec); // 恢复具有额外最后一个维度（表示实部和虚部）的实数张量
    // spec[:, :, :, 0] #实部
    // spec[:, :, :, 1] #虚部
    auto mag = torch::sqrt(stft_spec.pow(2).sum(-1) + (1e-9));
    auto pha = torch::atan2(stft_spec.index({"...", 1}), stft_spec.index({"...", 0}) + (1e-5));
    mag = torch::pow(mag, compress_factor);
    auto com = torch::stack((mag * torch::cos(pha), mag * torch::sin(pha)), -1);
    return std::make_tuple<>(mag, pha, com);
}

static torch::Tensor mag_pha_istft(
    torch::Tensor mag,
    torch::Tensor pha,
    float compress_factor = 0.3
) {
    mag = torch::pow(mag, (1.0 / compress_factor));
    auto com = torch::complex(mag * torch::cos(pha), mag * torch::sin(pha));
    auto window = torch::hann_window(win_size);
    return torch::istft(com, n_fft, hop_size, win_size, window, true);
}

static void denoise() {
    // torch::stft()
}

int main() {
    std::ifstream input;
    std::ofstream output;
    input .open("D:/tmp/noise.pcm",   std::ios_base::binary);
    output.open("D:/tmp/denoise.pcm", std::ios_base::binary | std::ios_base::app);
    int seconds = 1;
    int size = 16000;
    std::vector<short> data;
    std::vector<float> data_f;
    data.resize(size);
    data_f.resize(size);
    auto model = torch::jit::load("D:/download/model.pt");
    while(input.read(reinterpret_cast<char*>(data.data()), size * sizeof(short))) {
        std::copy_n(data.data(), size, data_f.data());
        auto noisy_wav = torch::from_blob(data_f.data(), {1, size}, torch::kFloat32);
        noisy_wav = noisy_wav / 32768.0F;
        auto norm_factor = torch::sqrt(noisy_wav.sizes()[1] / torch::sum(noisy_wav.pow(2.0)));
        auto noisy_audio = (noisy_wav * norm_factor);
        auto tuple = mag_pha_stft(noisy_audio);

        std::cout << std::get<0>(tuple).sizes() << '\n';
        std::cout << std::get<1>(tuple).sizes() << '\n';

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(std::get<0>(tuple));
        inputs.push_back(std::get<1>(tuple));
        auto o = model.forward(inputs);
        auto t = o.toTuple();
        auto result = mag_pha_istft(t->elements()[0].toTensor(), t->elements()[1].toTensor());

        // auto result = mag_pha_istft(std::get<0>(tuple), std::get<1>(tuple));

        result = result / norm_factor;
        result = result * 32768.0F;
        float* data_f_ptr = reinterpret_cast<float*>(result.data_ptr());
        std::copy_n(data_f_ptr, size, data.data());
        // std::memcpy(data_f.data(), result.data_ptr(), size * sizeof(float));
        // std::copy_n(data_f.data(), size, data.data());
        output.write(reinterpret_cast<char*>(data.data()), size * sizeof(short));
        output.flush();
    }
    input.close();
    output.close();
    std::cout << "完成\n";
    return 0;
}

/*

import torch
import torch.nn as nn

from modelscope.fileio import File
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.audio.ans.zipenhancer import ZipEnhancer

class OnnxModel(nn.Module):

    def __init__(self, ans):
        super().__init__()
        self.model = ans.model.model

    def forward(self, noisy_amp, noisy_pha):
        amp_g, pha_g, _, _, _ = self.model(noisy_amp, noisy_pha)
        return amp_g, pha_g

ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model = 'damo/speech_zipenhancer_ans_multiloss_16k_base'
)

model = OnnxModel(ans)

seconds = 1
f = 201
t = 160 * seconds + 1
noisy_mag, noisy_pha = torch.randn(1, f, t), torch.randn(1, f, t)
model = torch.jit.trace(model, (noisy_mag, noisy_pha))
model.eval()

print(model)

torch.jit.save(model, 'model.pt')

*/