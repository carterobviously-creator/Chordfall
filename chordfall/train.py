import os
import json
import time
import numpy as np
from tqdm import trange

import torch
import torch.nn.functional as F

from chordfall.model import TinyWaveRNN
from chordfall.mulaw import mu_law_encode
from chordfall.synth import make_synth_clip


def _select_device(prefer: str = "auto") -> str:
    prefer = (prefer or "auto").lower()

    if prefer == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "GPU training requested (cuda) but CUDA is not available. "
            "Install a CUDA-enabled PyTorch build and NVIDIA drivers."
        )

    if prefer == "cpu":
        return "cpu"

    # auto
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_model(
    out_dir: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    sample_rate: int,
    seconds_per_example: float,
    lr: float,
    hidden: int,
    layers: int,
    seed: int = 0,
    device_preference: str = "auto",
):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")
    meta_path = os.path.join(out_dir, "meta.json")
    ckpt_path = os.path.join(out_dir, "model.pt")

    if seed == 0:
        seed = int(time.time()) % 2_000_000_000
    rng = np.random.default_rng(seed)

    device = _select_device(device_preference)

    model = TinyWaveRNN(vocab_size=256, hidden=hidden, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "device_preference": device_preference,
        "seed": seed,
        "sample_rate": sample_rate,
        "seconds_per_example": seconds_per_example,
        "train": {"steps": steps, "batch_size": batch_size, "seq_len": seq_len, "lr": lr},
        "model": {"type": "TinyWaveRNN", "hidden": hidden, "layers": layers, "vocab_size": 256},
        "notes": "Trained from scratch on synthetic chord/tone audio using 8-bit mu-law tokens.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    def sample_batch():
        needed = batch_size * (seq_len + 1)
        buf = []
        total = 0
        while total < needed:
            clip = make_synth_clip(sample_rate, seconds_per_example, rng)
            tok = mu_law_encode(clip)
            buf.append(tok)
            total += tok.shape[0]
        stream = np.concatenate(buf, axis=0)[:needed]
        stream = stream.reshape(batch_size, seq_len + 1)
        x = stream[:, :seq_len]
        y = stream[:, 1:]
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()

    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"device={device} amp={use_amp}\n")
        lf.flush()

        for step in trange(1, steps + 1, desc="training"):
            model.train()
            x, y = sample_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if step % 50 == 0 or step == 1:
                msg = f"step={step} loss={loss.item():.4f}"
                lf.write(msg + "\n")
                lf.flush()

            if step % 500 == 0:
                torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)

    torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)
    return meta_path, ckpt_path, log_path
