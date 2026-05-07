# Chordfall

A tiny **from-scratch** audio generation prototype + **DSP vocoder** built with **Gradio**.

## What’s included
- **Train tab**: trains a tiny GRU waveform model from scratch on *synthetic chord/tone audio* (no dataset files needed)
- **Generate tab**: generates WAV audio from a trained checkpoint
- **Vocode tab**: classic multi-band channel vocoder (DSP) for lyrics/voice over a carrier

## Quickstart (Windows)
1. Install Python 3.10+.
2. Double click `run.bat`.
3. Open the Gradio link in your browser.

## GPU training (NVIDIA)
By default Chordfall uses `auto` device (GPU if available). To force GPU, set **Device = cuda** in the Train tab.

Important: `pip install torch` usually installs a **CPU-only** build on Windows. For CUDA you must install the CUDA-enabled PyTorch build from:
https://pytorch.org/get-started/locally/

## Tips
- Start with: sample rate 16000, steps 1500–5000.
- Vocoder: record voice as modulator, use generated WAV as carrier. Ensure both have the same sample rate.

## Output
- Training runs are saved under `runs/<timestamp>/model.pt`
- Generated WAVs are saved under `runs/_generated/`
