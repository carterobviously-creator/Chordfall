import numpy as np


def softclip(x, drive=1.0):
    return np.tanh(x * drive)


def _env(n, sr, a=0.005, d=0.08, s=0.6, r=0.08):
    env = np.ones(n, dtype=np.float32) * s
    aN = int(a * sr)
    dN = int(d * sr)
    rN = int(r * sr)
    if aN > 0:
        env[:aN] = np.linspace(0, 1, aN, dtype=np.float32)
    if dN > 0 and aN + dN < n:
        env[aN:aN + dN] = np.linspace(1, s, dN, dtype=np.float32)
    if rN > 0:
        env[-rN:] = np.linspace(s, 0, rN, dtype=np.float32)
    return env


def _kick(sr, n, rng):
    t = np.arange(n, dtype=np.float32) / sr
    f0 = float(rng.uniform(90, 140))
    f1 = float(rng.uniform(35, 60))
    # exponential pitch drop
    f = f1 + (f0 - f1) * np.exp(-t * 18.0)
    phase = 2 * np.pi * np.cumsum(f) / sr
    x = np.sin(phase)
    x *= np.exp(-t * 12.0)
    x = softclip(x, 2.5)
    return x.astype(np.float32)


def _snare(sr, n, rng):
    t = np.arange(n, dtype=np.float32) / sr
    noise = rng.normal(0, 1, size=n).astype(np.float32)
    tone = np.sin(2 * np.pi * float(rng.uniform(160, 220)) * t).astype(np.float32)
    x = 0.8 * noise + 0.2 * tone
    x *= np.exp(-t * 18.0)
    return softclip(x, 1.6).astype(np.float32)


def _hat(sr, n, rng):
    t = np.arange(n, dtype=np.float32) / sr
    noise = rng.normal(0, 1, size=n).astype(np.float32)
    x = noise * np.exp(-t * float(rng.uniform(45, 80)))
    return softclip(x, 1.2).astype(np.float32)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def chord_midi(root_midi: int, quality: str):
    if quality == "maj":
        intervals = [0, 4, 7]
    elif quality == "min":
        intervals = [0, 3, 7]
    elif quality == "dim":
        intervals = [0, 3, 6]
    elif quality == "sus2":
        intervals = [0, 2, 7]
    elif quality == "sus4":
        intervals = [0, 5, 7]
    else:
        intervals = [0, 4, 7]
    return [root_midi + i for i in intervals]


def make_synth_music_clip(sr: int, seconds: float, rng: np.random.Generator, genre: str = "mix") -> np.ndarray:
    """Procedural 'music-ish' clip: drums + bass + chords.

    No external dataset required.
    """
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / sr

    # genre-ish tempo ranges
    genre = (genre or "mix").lower()
    if genre == "dnb":
        bpm = float(rng.uniform(160, 180))
    elif genre == "house":
        bpm = float(rng.uniform(120, 130))
    elif genre == "trap":
        bpm = float(rng.uniform(130, 155))
    elif genre == "lofi":
        bpm = float(rng.uniform(70, 95))
    elif genre == "ambient":
        bpm = float(rng.uniform(50, 80))
    else:
        bpm = float(rng.uniform(80, 170))

    beat = 60.0 / bpm
    step = beat / 4.0  # 16th note grid
    stepN = max(1, int(step * sr))

    # harmonic plan
    qualities = ["maj", "min", "sus2", "sus4"]
    base_root = int(rng.integers(45, 60))  # ~A2..B3
    prog = [
        (base_root + 0, qualities[int(rng.integers(0, len(qualities)))]),
        (base_root + int(rng.integers(-5, 6)), qualities[int(rng.integers(0, len(qualities)))]),
        (base_root + int(rng.integers(-7, 8)), qualities[int(rng.integers(0, len(qualities)))]),
        (base_root + int(rng.integers(-5, 6)), qualities[int(rng.integers(0, len(qualities)))]),
    ]

    out = np.zeros(n, dtype=np.float32)

    # drums
    for idx in range(0, n, stepN):
        s = idx
        e = min(n, idx + stepN)
        localN = e - s

        step_i = idx // stepN
        pos16 = step_i % 16

        # house: 4-on-the-floor
        if genre in ("house", "mix") and pos16 % 4 == 0:
            out[s:e] += 0.8 * _kick(sr, localN, rng)

        # trap-ish: sparse kick
        if genre == "trap" and (pos16 in (0, 7, 8, 15)) and rng.random() < 0.7:
            out[s:e] += 0.9 * _kick(sr, localN, rng)

        # dnb-ish: kick on 1 and 11
        if genre == "dnb" and pos16 in (0, 10) and rng.random() < 0.9:
            out[s:e] += 0.9 * _kick(sr, localN, rng)

        # snare on 2 and 4 (backbeat)
        if pos16 in (4, 12) and rng.random() < (0.9 if genre != "ambient" else 0.2):
            out[s:e] += 0.6 * _snare(sr, localN, rng)

        # hats on off-steps
        hat_prob = 0.6
        if genre == "trap":
            hat_prob = 0.85
        if genre == "ambient":
            hat_prob = 0.15
        if (pos16 % 2 == 1) and rng.random() < hat_prob:
            out[s:e] += 0.25 * _hat(sr, localN, rng)

    # bass + chords in longer blocks
    blocks = max(1, int(seconds / (beat * 4)))  # roughly per bar
    blockN = max(1, n // blocks)

    for b in range(blocks):
        root, q = prog[b % len(prog)]
        s = b * blockN
        e = n if b == blocks - 1 else (b + 1) * blockN
        segN = e - s
        seg_t = t[:segN]

        # bass
        bass_m = root - 12
        fb = midi_to_hz(bass_m)
        bass = 0.35 * np.sin(2 * np.pi * fb * seg_t)
        bass *= _env(segN, sr, a=0.002, d=0.06, s=0.7, r=0.05)

        # chord stabs (additive)
        notes = chord_midi(root, q)
        chord = np.zeros(segN, dtype=np.float32)
        for m in notes:
            f = midi_to_hz(m)
            chord += 0.22 * np.sin(2 * np.pi * f * seg_t)
            chord += 0.08 * np.sin(2 * np.pi * (2 * f) * seg_t)
        chord *= _env(segN, sr, a=0.005, d=0.12, s=0.4, r=0.08)

        out[s:e] += bass.astype(np.float32) + chord.astype(np.float32)

    # gentle noise + saturation
    out += (rng.normal(0, 0.003, size=n)).astype(np.float32)
    out = softclip(out, 1.5)

    mx = float(np.max(np.abs(out)) + 1e-8)
    out = (out / mx).astype(np.float32)
    return out
