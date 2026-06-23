"""Low-memory settings for CPU / free-tier hosting (e.g. Render)."""

import gc
import os


def is_low_memory_mode() -> bool:
    return os.environ.get("LOW_MEMORY", "").lower() in ("1", "true", "yes") or os.environ.get(
        "FORCE_CPU", ""
    ).lower() in ("1", "true", "yes")


def configure_torch():
    """Call once at process startup before loading the model."""
    try:
        import torch
    except ImportError:
        return

    threads = int(os.environ.get("TORCH_NUM_THREADS", "1" if is_low_memory_mode() else "2"))
    torch.set_num_threads(max(1, threads))
    torch.set_grad_enabled(False)

    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


def release_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
