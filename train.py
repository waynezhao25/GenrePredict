import os
os.add_dll_directory(r"C:\Users\Ninja\miniconda3\envs\ffmpegdlls\Library\bin")
os.add_dll_directory(r"C:\Users\Ninja\miniconda3\envs\ffmpegdlls\DLLs")

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from config import GENRES, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
from pathlib import Path

# Data cleaning
df = pd.read_csv("data/metadata/tracks.csv", header=[0,1], low_memory =False)
print(df.columns)

tracks = df[[('track', 'genre_top'), ('set', 'split')]].copy()
tracks.columns = ['genre_top', 'split']
tracks = tracks.dropna(subset=['genre_top'])
tracks = tracks[tracks['genre_top'].isin(GENRES)]
print(tracks['genre_top'].value_counts())
print(tracks['split'].value_counts())

# Processing Audio
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)

def wav_to_mel(
    audio_path:str,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)

    # check stereo, converts to mono
    if waveform.shape[0] > 1: 
        waveform = waveform.mean(dim = 0, keepdim = True)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power = 2
    )

    amplitude_to_db = T.AmplitudeToDB(stype='power')
    mel_spec = mel_transform(waveform)
    log_mel_spec = amplitude_to_db(mel_spec)

    return log_mel_spec

log_mel = wav_to_mel("/data/training_data/000/000002.wav")
print(f"{log_mel.shape}")
print(f"{log_mel.min():.2f}, {log_mel.max():.2f}")
print(mel_transform)