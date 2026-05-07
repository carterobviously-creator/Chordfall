# Chordfall datasets

Chordfall can train without any dataset (synthetic generator), but you can get better results by training on small legal datasets.

This repo **does not include real songs** to avoid copyright and repo-size issues.

## Option 1: Procedural drum + chord dataset (built-in)
No downloads. Use the **Synthetic (music)** source in the Train tab.

## Option 2: NSynth (instrument notes)
NSynth is a large dataset of individual instrument notes (not full songs). It’s useful for timbre.

### Download (recommended small subset)
Use the downloader script which grabs the official NSynth TFRecord archive and extracts a **small subset** of audio files into `datasets/nsynth/`.

> Disk: You control how many files are extracted via `--limit`.

```bat
.venv\Scripts\python scripts\download_nsynth.py --out datasets\nsynth --limit 2000
```

Then set Train → **Source = Dataset folder** and choose `datasets/nsynth/audio`.

## Option 3: Small drum one-shots pack
Use the included generator (no downloads) for now. If you want external drum packs, tell us the license/source and we can add a downloader.
