import numpy as np
import torch

from chordfall.model import TinyWaveRNN
from chordfall.mulaw import mu_law_decode


def _sample_from_logits(logits: torch.Tensor, temperature: float, rng: np.random.Generator) -> int:
    logits = logits.float()
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1).detach().cpu().numpy()
    return int(rng.choice(len(probs), p=probs))


def generate_wav(
    ckpt_path: str,
    seconds: float,
    sample_rate: int,
    temperature: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    if seed == 0:
        seed = int(np.random.randint(1, 2_000_000_000))
    rng = np.random.default_rng(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})
    mconf = (meta.get("model") or {})
    hidden = int(mconf.get("hidden", 128))
    layers = int(mconf.get("layers", 2))

    model = TinyWaveRNN(vocab_size=256, hidden=hidden, layers=layers).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n = int(seconds * sample_rate)

    x = torch.tensor([[128]], dtype=torch.long, device=device)
    h = None
    out_tokens = np.zeros(n, dtype=np.int64)

    with torch.no_grad():
        for i in range(n):
            logits, h = model(x, h)
            next_logits = logits[0, -1]
            tok = _sample_from_logits(next_logits, temperature=temperature, rng=rng)
            out_tokens[i] = tok
            x = torch.tensor([[tok]], dtype=torch.long, device=device)

    wav = mu_law_decode(out_tokens).astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    return wav
