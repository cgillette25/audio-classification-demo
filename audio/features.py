import torch
import torchaudio

class LogMelSpec(torch.nn.Module):
    """
    Converts waveform -> log-mel spectrogram.
    Output shape: (n_mels, time)
    """
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 1024,
        n_mels: int = 128,
        f_min: int = 20,
        f_max: int = 14000,
        power: float = 2.0,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
        )
        self.eps = eps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (1, T) expected
        m = self.melspec(waveform)  # (1, n_mels, time)
        m = torch.log(m + self.eps)
        return m.squeeze(0)         # (n_mels, time)
