---
title: Audio Classification Demo (ESC-50) — Log-Mel + CNN
emoji: 🔊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Audio Classification Demo (ESC-50) — Log-Mel Spectrogram + CNN

## Overview
This project is an end-to-end audio ML demo designed for entertainment and media workflows such as content tagging, sound library organization, and automated QA. It covers:
- Signal processing frontend (waveform → STFT → mel scaling → log compression)
- PyTorch modeling (CNN classifier)
- Reproducible training + evaluation (accuracy, macro F1, confusion matrix)
- Interactive deployment (Gradio UI on Hugging Face Spaces)

## Why this is relevant to media companies
Audio classification is a core building block for:
- **Content tagging** (automated metadata for large audio catalogs)
- **Pipeline QA** (detect unexpected noise/classes in assets)
- **Search and retrieval** (fast indexing by sound type)
- **Real-time inference constraints** (short fixed-window processing)

## Dataset
**ESC-50**: 2,000 environmental audio recordings across 50 classes, organized into 5 folds for cross-validation.

Expected folder structure:
data/ESC-50/
audio/
meta/esc50.csv


## Method
### Feature extraction (torchaudio)
1. Resample to a fixed sample rate (default: 32 kHz)
2. Convert to mono
3. Pad/trim to a fixed clip duration (default: 5 seconds)
4. Compute **mel spectrogram**
5. Apply **log compression**
6. Normalize per clip

### Model
A compact CNN operating on log-mel spectrograms:
- Conv → BN → ReLU → Pool (repeated)
- Global average pooling
- Linear classification head

This baseline is intentionally:
- Fast for inference
- Stable to train
- Easy to deploy

## Results
After training, the evaluation script reports:
- Accuracy
- Macro F1
- Per-class precision/recall/F1
- Confusion matrix (saved for notebook visualization)

## Run locally

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
