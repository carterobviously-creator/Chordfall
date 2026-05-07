import os


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= 1024.0
        if n < 1024:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"
