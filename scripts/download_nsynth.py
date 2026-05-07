import argparse
import os
import io
import sys
import zipfile
import urllib.request
import shutil


NSYNTH_URL = "https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-test.jsonwav.tar.gz"


def _download(url: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        total = r.length
        read = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                pct = 100.0 * read / total
                print(f"  {pct:5.1f}%", end="\r")
    print("\nDone.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory, e.g. datasets/nsynth")
    ap.add_argument("--limit", type=int, default=2000, help="Max number of wav files to extract")
    args = ap.parse_args()

    out_dir = args.out
    tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    archive_path = os.path.join(tmp_dir, "nsynth-test.jsonwav.tar.gz")

    if not os.path.exists(archive_path):
        _download(NSYNTH_URL, archive_path)

    # Extract tar.gz without extra deps (uses shutil.unpack_archive)
    unpack_dir = os.path.join(tmp_dir, "unpack")
    if os.path.isdir(unpack_dir):
        shutil.rmtree(unpack_dir)
    os.makedirs(unpack_dir, exist_ok=True)

    print("Unpacking archive...")
    shutil.unpack_archive(archive_path, unpack_dir)

    # nsynth-test/audio/*.wav
    audio_src = os.path.join(unpack_dir, "nsynth-test", "audio")
    if not os.path.isdir(audio_src):
        raise SystemExit(f"Could not find audio directory at: {audio_src}")

    audio_out = os.path.join(out_dir, "audio")
    os.makedirs(audio_out, exist_ok=True)

    wavs = [f for f in os.listdir(audio_src) if f.lower().endswith(".wav")]
    wavs.sort()

    limit = min(args.limit, len(wavs))
    print(f"Copying {limit} wav files to {audio_out} ...")

    for i in range(limit):
        shutil.copy2(os.path.join(audio_src, wavs[i]), os.path.join(audio_out, wavs[i]))

    print("Done.")
    print(f"Now you can train with Source=Dataset folder and path={audio_out}")


if __name__ == "__main__":
    main()
