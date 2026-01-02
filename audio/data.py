from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
import torchaudio

from .features import LogMelSpec
from .utils import pad_or_trim

@dataclass
class AudioExample:
    x: torch.Tensor
    y: int
    path: str

class ESC50Dataset(torch.utils.data.Dataset):
    """
    ESC-50 expects:
      root/
        audio/*.wav
        meta/esc50.csv
    """
    def __init__(
        self,
        root: str,
        sample_rate: int,
        clip_seconds: float,
        features: LogMelSpec,
        fold: int,
        split: str,  # "train" or "test"
        augment: bool = False,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_seconds)
        self.features = features

        # SpecAugment-style masks (applied on log-mel, train only)
        self.augment = augment
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=48)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)

        meta = pd.read_csv(self.root / "meta" / "esc50.csv")
        # ESC-50 uses 5 folds
        if split == "test":
            meta = meta[meta["fold"] == fold]
        else:
            meta = meta[meta["fold"] != fold]

        self.meta = meta.reset_index(drop=True)
        self.labels = sorted(self.meta["category"].unique().tolist())
        self.label_to_idx = {c: i for i, c in enumerate(self.labels)}

        self.resampler = None  # lazily init if needed

    def __len__(self):
        return len(self.meta)

    def _ensure_sr(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return waveform
        if self.resampler is None or self.resampler.orig_freq != sr:
            self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        return self.resampler(waveform)

    def __getitem__(self, idx: int) -> AudioExample:
        row = self.meta.iloc[idx]
        wav_path = self.root / "audio" / row["filename"]

        waveform, sr = torchaudio.load(str(wav_path))  # waveform: (C, T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
        waveform = self._ensure_sr(waveform, sr)
        waveform = pad_or_trim(waveform, self.clip_len)

        feat = self.features(waveform)  # (n_mels, time)

        # Normalize per-mel bin across time
        feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-6)

        # SpecAugment (train only)
        if self.augment:
            f = feat.unsqueeze(0)  # (1, n_mels, time)
            f = self.time_mask(f)
            f = self.freq_mask(f)
            feat = f.squeeze(0)

        y = self.label_to_idx[row["category"]]
        return AudioExample(x=feat, y=y, path=str(wav_path))
