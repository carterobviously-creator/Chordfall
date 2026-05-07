import numpy as np
from scipy.signal import butter, lfilter


def _bandpass(sr, low, high, order=4):
    nyq = 0.5 * sr
    lowc = max(1.0, low) / nyq
    highc = min(nyq - 1.0, high) / nyq
    b, a = butter(order, [lowc, highc], btype="band")
    return b, a


def _lowpass(sr, cutoff, order=2):
    nyq = 0.5 * sr
    c = min(nyq - 1.0, cutoff) / nyq
    b, a = butter(order, c, btype="low")
    return b, a


def vocode(
    carrier: np.ndarray,
    modulator: np.ndarray,
    sr: int,
    bands: int = 16,
    fmin: float = 80.0,
    fmax: float = 6000.0,
    envelope_lp_hz: float = 30.0,
) -> np.ndarray:
    """Classic channel vocoder (DSP).

    - Split modulator into bands, extract amplitude envelope per band.
    - Split carrier into bands, multiply by modulator envelope.
    - Sum bands.
    """
    carrier = carrier.astype(np.float32)
    modulator = modulator.astype(np.float32)

    n = min(len(carrier), len(modulator))
    carrier = carrier[:n]
    modulator = modulator[:n]

    edges = np.geomspace(fmin, fmax, bands + 1)
    out = np.zeros(n, dtype=np.float32)

    b_lp, a_lp = _lowpass(sr, envelope_lp_hz, order=2)

    for i in range(bands):
        lo = float(edges[i])
        hi = float(edges[i + 1])

        b_bp, a_bp = _bandpass(sr, lo, hi, order=4)

        m_band = lfilter(b_bp, a_bp, modulator)
        c_band = lfilter(b_bp, a_bp, carrier)

        env = np.abs(m_band)
        env = lfilter(b_lp, a_lp, env)

        out += c_band * env

    mx = float(np.max(np.abs(out)) + 1e-8)
    out = out / mx
    out = np.tanh(out * 1.2)
    return out.astype(np.float32)
