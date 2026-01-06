import os
from pathlib import Path
from urllib.request import urlretrieve
import torch
import numpy as np
import gradio as gr
import torchaudio

from audio.features import LogMelSpec
from audio.model import SimpleAudioCNN
from audio.utils import pad_or_trim

# Load checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_PATH = Path("checkpoints/esc50_cnn.pt")
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_URL = "https://raw.githubusercontent.com/cgillette25/audio-classification-demo/main/assets/checkpoints/esc50_cnn.pt"

def ensure_checkpoint():
    if CKPT_PATH.exists():
        return
    print(f"Downloading checkpoint from {MODEL_URL}")
    urlretrieve(MODEL_URL, CKPT_PATH)

ensure_checkpoint()

ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)
cfg = ckpt["cfg"]
LABELS = ckpt["labels"]


# Rebuild feature pipeline
feat = LogMelSpec(
    sample_rate=cfg["dataset"]["sample_rate"],
    n_fft=cfg["features"]["n_fft"],
    hop_length=cfg["features"]["hop_length"],
    win_length=cfg["features"]["win_length"],
    n_mels=cfg["features"]["n_mels"],
    f_min=cfg["features"]["f_min"],
    f_max=cfg["features"]["f_max"],
)

model = SimpleAudioCNN(n_classes=len(LABELS), dropout=cfg["model"]["dropout"])
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

_resampler = None

def to_mono_resample(waveform, sr, target_sr):
    global _resampler
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        if _resampler is None or _resampler.orig_freq != sr:
            _resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = _resampler(waveform)
    return waveform


def predict(audio_file):
    if audio_file is None:
        return None, {}

    # gradio returns filepath
    waveform, sr = torchaudio.load(audio_file)
    waveform = to_mono_resample(waveform, sr, cfg["dataset"]["sample_rate"])

    clip_len = int(cfg["dataset"]["sample_rate"] * cfg["dataset"]["clip_seconds"])
    waveform = pad_or_trim(waveform, clip_len)

    m = feat(waveform)  # (n_mels, time)
    m = (m - m.mean()) / (m.std() + 1e-6)

    x = m.unsqueeze(0).to(DEVICE)  # (1, n_mels, time)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    topk = probs.argsort()[::-1][:5]
    pred_label = LABELS[int(topk[0])]

    # Return spectrogram as image-like array for display (normalize 0..1)
    spec = m.cpu().numpy()
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)

    conf = {LABELS[i]: float(probs[i]) for i in topk}
    return spec_norm, conf

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload an audio clip (wav/mp3)"),
    outputs=[
        gr.Image(label="Log-Mel Spectrogram", type="numpy"),
        gr.Label(label="Top Predictions"),
    ],
    title="Audio Classification Demo (ESC-50) — Log-Mel + CNN",
    description=(
        "Upload a clip to classify environmental sounds. "
        "This demo uses a log-mel spectrogram frontend (torchaudio) and a CNN trained on ESC-50."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

