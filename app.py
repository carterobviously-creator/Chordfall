import os
import time
import json
import numpy as np
import soundfile as sf
import gradio as gr

from chordfall.train import train_model
from chordfall.generate import generate_wav
from chordfall.util import ensure_dir, human_bytes
from chordfall.vocoder import vocode

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
ensure_dir(RUNS_DIR)


def list_checkpoints():
    if not os.path.isdir(RUNS_DIR):
        return []
    items = []
    for root, _, files in os.walk(RUNS_DIR):
        for f in files:
            if f.endswith(".pt"):
                items.append(os.path.join(root, f))
    items.sort(reverse=True)
    return items


def do_train(steps, batch_size, seq_len, sr, seconds_per_example, lr, hidden, layers, seed, device_pref, source, genre, dataset_folder):
    steps = int(steps)
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    sr = int(sr)
    seconds_per_example = float(seconds_per_example)
    lr = float(lr)
    hidden = int(hidden)
    layers = int(layers)
    seed = int(seed)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    meta_path, ckpt_path, log_path = train_model(
        out_dir=out_dir,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        sample_rate=sr,
        seconds_per_example=seconds_per_example,
        lr=lr,
        hidden=hidden,
        layers=layers,
        seed=seed,
        device_preference=device_pref,
        source=source,
        genre=genre,
        dataset_folder=(dataset_folder or None),
    )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    summary = (
        f"Run: {run_id}\n"
        f"Checkpoint: {ckpt_path}\n"
        f"Meta: {meta_path}\n"
        f"Log: {log_path}\n"
        f"Device: {meta.get('device')}\n"
        f"Source: {meta.get('source')} genre={meta.get('genre')}\n"
        f"Dataset WAVs: {meta.get('dataset_wav_count')}\n"
        f"Params: {meta['model']}\n"
    )
    return summary, ckpt_path


def do_generate(ckpt_path, seconds, sr, temperature, seed, out_name):
    seconds = float(seconds)
    sr = int(sr)
    temperature = float(temperature)
    seed = int(seed)
    out_name = (out_name or "sample").strip()
    if not out_name.endswith(".wav"):
        out_name += ".wav"

    out_dir = os.path.join(RUNS_DIR, "_generated")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, out_name)

    wav = generate_wav(
        ckpt_path=ckpt_path,
        seconds=seconds,
        sample_rate=sr,
        temperature=temperature,
        seed=seed,
    )
    sf.write(out_path, wav, sr)

    info = f"Wrote: {out_path} ({human_bytes(os.path.getsize(out_path))})"
    return out_path, info


with gr.Blocks(title="Chordfall (Prototype)") as demo:
    gr.Markdown(
        "# Chordfall (Prototype)\n"
        "A tiny from-scratch audio generator + DSP vocoder.\n\n"
        "- Can train without datasets (procedural music synth).\n"
        "- Optional: train on a folder of WAVs (ex: NSynth subset).\n"
    )

    with gr.Tab("Train"):
        device_pref = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
        source = gr.Dropdown(
            ["synthetic_music", "synthetic_basic", "dataset", "mixed"],
            value="synthetic_music",
            label="Source",
        )
        genre = gr.Dropdown(["mix", "house", "trap", "lofi", "dnb", "ambient"], value="mix", label="Synthetic genre")
        dataset_folder = gr.Textbox(value="datasets/nsynth/audio", label="Dataset folder (WAVs) when Source=dataset/mixed")

        steps = gr.Slider(100, 20000, value=2000, step=50, label="Training steps")
        batch_size = gr.Slider(1, 64, value=8, step=1, label="Batch size")
        seq_len = gr.Slider(128, 4096, value=1024, step=128, label="Sequence length (samples)")
        sr = gr.Dropdown([8000, 12000, 16000, 22050], value=16000, label="Sample rate")
        seconds_per_example = gr.Slider(0.25, 8.0, value=2.0, step=0.25, label="Clip length (seconds)")
        lr = gr.Slider(1e-5, 5e-3, value=5e-4, step=1e-5, label="Learning rate")
        hidden = gr.Slider(32, 512, value=128, step=32, label="Hidden size")
        layers = gr.Slider(1, 4, value=2, step=1, label="GRU layers")
        seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")

        train_btn = gr.Button("Train")
        train_out = gr.Textbox(label="Training result", lines=10)
        ckpt_out = gr.Textbox(label="Checkpoint path")

        train_btn.click(
            fn=do_train,
            inputs=[steps, batch_size, seq_len, sr, seconds_per_example, lr, hidden, layers, seed, device_pref, source, genre, dataset_folder],
            outputs=[train_out, ckpt_out],
        )

        gr.Markdown(
            "### Download NSynth subset (optional)\n"
            "Run this in a terminal (after run.bat made the venv):\n\n"
            "```bat\n"
            ".venv\\Scripts\\python scripts\\download_nsynth.py --out datasets\\nsynth --limit 2000\n"
            "```\n"
        )

    with gr.Tab("Generate"):
        ckpt_path = gr.Textbox(label="Checkpoint path", value="")
        refresh = gr.Button("Refresh list of checkpoints")
        ckpt_list = gr.Dropdown(choices=list_checkpoints(), label="Pick a checkpoint")
        refresh.click(lambda: gr.Dropdown(choices=list_checkpoints()), outputs=[ckpt_list])

        def pick_ckpt(x):
            return x

        ckpt_list.change(pick_ckpt, inputs=[ckpt_list], outputs=[ckpt_path])

        seconds = gr.Slider(0.5, 20.0, value=6.0, step=0.5, label="Seconds to generate")
        sr2 = gr.Dropdown([8000, 12000, 16000, 22050], value=16000, label="Sample rate")
        temperature = gr.Slider(0.6, 1.5, value=1.0, step=0.05, label="Temperature")
        seed2 = gr.Number(value=0, precision=0, label="Seed (0 = random)")
        out_name = gr.Textbox(label="Output file name", value="sample.wav")

        gen_btn = gr.Button("Generate WAV")
        audio_out = gr.Audio(label="Output audio", type="filepath")
        gen_info = gr.Textbox(label="Info", lines=2)

        gen_btn.click(
            fn=do_generate,
            inputs=[ckpt_path, seconds, sr2, temperature, seed2, out_name],
            outputs=[audio_out, gen_info],
        )

    with gr.Tab("Vocode (Lyrics/Voice)"):
        gr.Markdown(
            "This is a **DSP vocoder** (no pretrained model).\n\n"
            "Provide:\n"
            "- **Carrier**: music/synth audio (you can use a generated WAV).\n"
            "- **Modulator**: voice audio (record or upload). Speak/sing lyrics here.\n\n"
            "Tip: make sure both files use the same sample rate."
        )

        carrier_in = gr.Audio(label="Carrier (music/synth)", type="numpy")
        mod_in = gr.Audio(label="Modulator (voice)", type="numpy")

        bands = gr.Slider(8, 48, value=16, step=1, label="Bands")
        env_lp = gr.Slider(5, 80, value=30, step=1, label="Envelope lowpass (Hz)")

        voc_btn = gr.Button("Vocode")
        voc_out = gr.Audio(label="Vocoded output", type="numpy")

        def do_vocode(carrier, mod, bands, env_lp):
            if carrier is None or mod is None:
                raise gr.Error("Please provide both carrier and modulator audio.")
            sr_c, c = carrier
            sr_m, m = mod
            if sr_c != sr_m:
                raise gr.Error(f"Sample rates must match. Carrier SR={sr_c}, Modulator SR={sr_m}")

            if c.ndim == 2:
                c = c.mean(axis=1)
            if m.ndim == 2:
                m = m.mean(axis=1)

            y = vocode(c, m, sr=sr_c, bands=int(bands), envelope_lp_hz=float(env_lp))
            return (sr_c, y)

        voc_btn.click(do_vocode, inputs=[carrier_in, mod_in, bands, env_lp], outputs=[voc_out])


if __name__ == "__main__":
    demo.launch()
