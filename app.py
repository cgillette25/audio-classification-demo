import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import gradio as gr
import torchaudio

from audio.features import LogMelSpec
from audio.model import SimpleAudioCNN
from audio.utils import pad_or_trim

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Checkpoint download 
# -----------------------------
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

# -----------------------------
# Feature pipeline + model
# -----------------------------
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

# -----------------------------
# Audio helpers
# -----------------------------
_resampler = None

def to_mono_resample(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    global _resampler
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        if _resampler is None or _resampler.orig_freq != sr:
            _resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = _resampler(waveform)
    return waveform

def load_audio_from_gradio(audio_input):
    """
    Supports Gradio Audio in multiple formats:
      - filepath (str)
      - tuple: (sample_rate, numpy_array)
      - dict: {"path": "..."} or {"sampling_rate": sr, "data": array}
    Returns: waveform torch.Tensor shape (C, T), sr int
    """
    # filepath string
    if isinstance(audio_input, str):
        return torchaudio.load(audio_input)

    # dict format
    if isinstance(audio_input, dict):
        if audio_input.get("path"):
            return torchaudio.load(audio_input["path"])
        if "data" in audio_input and "sampling_rate" in audio_input:
            sr = int(audio_input["sampling_rate"])
            data = np.asarray(audio_input["data"], dtype=np.float32)
            # (T,) or (T, C) -> (C, T)
            if data.ndim == 1:
                data = data[:, None]
            waveform = torch.from_numpy(data.T)
            return waveform, sr

    # tuple(sr, data)
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        sr = int(audio_input[0])
        data = np.asarray(audio_input[1], dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        waveform = torch.from_numpy(data.T)
        return waveform, sr

    raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

# -----------------------------
# Inference
# -----------------------------
def predict(audio_input):
    if audio_input is None:
        return None, {}, "No audio provided."

    waveform, sr = load_audio_from_gradio(audio_input)
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

    # Return spectrogram array for display (normalize 0..1)
    spec = m.cpu().numpy()
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)

    conf = {LABELS[i]: float(probs[i]) for i in topk}
    return spec_norm, conf, f"Prediction: {pred_label}"

# -----------------------------
# UI
# -----------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(
        sources=["upload", "microphone"],
        type="numpy",
        label="Upload or record audio",
    ),
    outputs=[
        gr.Image(label="Log-Mel Spectrogram", type="numpy"),
        gr.Label(label="Top Predictions"),
        gr.Textbox(label="Predicted Class"),
    ],
    title="Audio Classification Demo (ESC-50) — Log-Mel + CNN",
    description=(
        "Upload or record a short clip to classify environmental sounds. "
        "This demo uses a log-mel spectrogram frontend (torchaudio) and a CNN trained on ESC-50."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
