from config import GENRES, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

import librosa
import librosa.display

glob('./data/training_data/*/*.wav')

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power = 2
    )

amplitude_to_db = T.AmplitudeToDB(stype='power')

# Data cleaning
df = pd.read_csv("data/metadata/tracks.csv", header=[0,1], low_memory =False)
 
tracks = df[[('track', 'genre_top'), ('set', 'split')]].copy()
tracks.columns = ['genre_top', 'split']
tracks = tracks.dropna(subset=['genre_top'])
tracks = tracks[tracks['genre_top'].isin(GENRES)]

# Processing Audio
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)

def wav_to_logmel(
    audio_path:str,
    mel_transform: T.MelSpectrogram,
    amplitude_to_db: T.AmplitudeToDB,
) -> torch.Tensor:
    waveform_np, sr = sf.read(audio_path, always_2d = True)
    waveform = torch.from_numpy(waveform_np.T).float()

    # check stereo, converts to mono
    if waveform.shape[0] > 1: 
        waveform = waveform.mean(dim = 0, keepdim = True)

    mel_spec = mel_transform(waveform)
    log_mel_spec = amplitude_to_db(mel_spec)

    return log_mel_spec

log_mel = wav_to_logmel("data/training_data/000/000002.wav", mel_transform, amplitude_to_db)
mel_img = log_mel.squeeze(0).numpy()

plt.figure(figsize=(10, 4))
plt.imshow(mel_img, origin="lower", aspect="auto", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel Spectrogram")
plt.xlabel("Time Frames")
plt.ylabel("Mel Frequency Bin")
plt.tight_layout()
plt.show()

print(f"{log_mel.shape}")
print(f"{log_mel.min():.2f}, {log_mel.max():.2f}")
print(mel_transform)

class GenreCNN (nn.Module):
    def __init__(self, num_classes):
        super(GenreCNN, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch_size, 1, n_mels, time_steps)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x