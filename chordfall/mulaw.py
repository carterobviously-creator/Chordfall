import numpy as np

MU = 255.0  # 8-bit mu-law (256 levels)


def mu_law_encode(x: np.ndarray, mu: float = MU) -> np.ndarray:
    # x in [-1, 1]
    x = np.clip(x, -1.0, 1.0)
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # map [-1,1] -> [0, mu]
    q = ((fx + 1.0) / 2.0 * mu + 0.5).astype(np.int64)
    return q


def mu_law_decode(q: np.ndarray, mu: float = MU) -> np.ndarray:
    # q in [0, mu]
    y = (q.astype(np.float32) / mu) * 2.0 - 1.0
    x = np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** np.abs(y) - 1.0)
    return x.astype(np.float32)
