# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Accompanying code for the notebook.
# We need to install matplotlib and jupyter notebook beforehand

import IPython.display as ipd
import matplotlib.pyplot as plt
import torch
import torchaudio


def plot_waveform_and_specgram(waveform, sample_rate, title):
    waveform = waveform.squeeze().detach().cpu().numpy()

    num_frames = waveform.shape[-1]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(time_axis, waveform, linewidth=1)
    ax1.grid(True)
    ax2.specgram(waveform, Fs=sample_rate)

    figure.suptitle(f"{title} - Waveform and specgram")
    plt.show()


def plot_waveform_and_melspec(waveform: torch.Tensor, sample_rate: int, title: str = "Audio"):
    """
    绘制波形和梅尔频谱图。
    参数：
        waveform: Tensor, shape = (1, num_samples) 或 (num_samples,)
        sample_rate: 采样率
        title: 图标题
    """
    waveform = waveform.squeeze().detach().cpu()
    time_axis = torch.arange(0, waveform.size(0)) / sample_rate

    # 生成 Mel 频谱
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )(waveform)

    # 转 dB（对数）形式
    mel_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spectrogram)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # 波形图
    ax1.plot(time_axis.numpy(), waveform.numpy(), linewidth=1)
    ax1.set_title(f"{title} - Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    # 梅尔频谱图
    ax2.imshow(mel_db.numpy(), aspect="auto", origin="lower", cmap="viridis",
               extent=[0, waveform.size(0) / sample_rate, 0, 80])
    ax2.set_title(f"{title} - Mel Spectrogram")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Mel Bin")

    plt.tight_layout()
    plt.show()

def play_audio(waveform, sample_rate):
    if waveform.dim() > 2:
        waveform = waveform.squeeze(0)
    waveform = waveform.detach().cpu().numpy()

    num_channels, *_ = waveform.shape
    if num_channels == 1:
        ipd.display(ipd.Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        ipd.display(ipd.Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")
