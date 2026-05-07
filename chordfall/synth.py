import numpy as np

NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}


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


def adsr_envelope(n, sr, a=0.01, d=0.08, s=0.7, r=0.1):
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


def make_synth_clip(sr: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / sr

    qualities = ["maj", "min", "sus2", "sus4", "dim"]
    num_chords = int(rng.integers(1, 5))
    chord_len = max(1, n // num_chords)

    base_root = int(rng.integers(48, 72))  # C3..B4
    clip = np.zeros(n, dtype=np.float32)

    for i in range(num_chords):
        root = base_root + int(rng.integers(-5, 6))
        q = qualities[int(rng.integers(0, len(qualities)))]
        notes = chord_midi(root, q)

        start = i * chord_len
        end = n if i == num_chords - 1 else (i + 1) * chord_len
        segN = end - start
        seg_t = t[:segN]

        seg = np.zeros(segN, dtype=np.float32)
        for m in notes:
            f = midi_to_hz(m)
            detune = 1.0 + float(rng.uniform(-0.003, 0.003))
            seg += 0.33 * np.sin(2 * np.pi * f * detune * seg_t)
            seg += 0.10 * np.sin(2 * np.pi * (2 * f) * seg_t)

        env = adsr_envelope(segN, sr)
        seg *= env
        clip[start:end] += seg

    clip += (rng.normal(0, 0.01, size=n)).astype(np.float32)
    clip = np.tanh(clip * 1.2)

    mx = float(np.max(np.abs(clip)) + 1e-8)
    clip = (clip / mx).astype(np.float32)
    return clip
