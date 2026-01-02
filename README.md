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
