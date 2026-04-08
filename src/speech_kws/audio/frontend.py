from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class AudioFrontend(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.kind = config["kind"]
        self.sample_rate = int(config.get("sample_rate", 16000))

        if self.kind == "log_mel":
            window_samples = int(self.sample_rate * float(config.get("window_ms", 30.0)) / 1000.0)
            hop_samples = int(self.sample_rate * float(config.get("hop_ms", 10.0)) / 1000.0)
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=window_samples,
                win_length=window_samples,
                hop_length=hop_samples,
                n_mels=int(config.get("n_mels", 40)),
                center=False,
                power=2.0,
            )
            self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
            self.feature_dim = int(config.get("n_mels", 40))
            self.expected_frames = int(config.get("expected_frames", 98))
        elif self.kind == "mfcc":
            window_samples = int(self.sample_rate * float(config.get("window_ms", 25.0)) / 1000.0)
            overlap_samples = int(self.sample_rate * float(config.get("overlap_ms", 10.0)) / 1000.0)
            hop_samples = max(1, window_samples - overlap_samples)
            n_mfcc = int(config.get("n_mfcc", 64))
            n_freqs = (window_samples // 2) + 1
            # Keep the default mel bank explicit and bounded by the FFT resolution.
            n_mels = int(config.get("n_mels", min(max(n_mfcc, 40), n_freqs)))
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=n_mfcc,
                log_mels=True,
                melkwargs={
                    "n_fft": window_samples,
                    "win_length": window_samples,
                    "hop_length": hop_samples,
                    "n_mels": n_mels,
                    "center": False,
                },
            )
            self.to_db = None
            self.feature_dim = n_mfcc
            self.expected_frames = int(config.get("expected_frames", 128))
        else:
            raise ValueError(f"Unsupported frontend kind: {self.kind}")

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=(-2, -1), keepdim=True)
        std = features.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        return (features - mean) / std

    def _pad_or_trim_time(self, features: torch.Tensor) -> torch.Tensor:
        time_dim = features.size(-1)
        if time_dim < self.expected_frames:
            features = F.pad(features, (0, self.expected_frames - time_dim))
        elif time_dim > self.expected_frames:
            features = features[..., : self.expected_frames]
        return features

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)
        mono = waveforms.squeeze(1)
        features = self.transform(mono)
        if self.to_db is not None:
            features = self.to_db(features.clamp(min=1e-10))
        features = self._pad_or_trim_time(features)
        return self._normalize(features)
