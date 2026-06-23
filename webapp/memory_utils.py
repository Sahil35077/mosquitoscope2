"""Low-memory settings for CPU / free-tier hosting (e.g. Render)."""

import gc
import os

_torch_configured = False


def is_low_memory_mode() -> bool:
    return os.environ.get("LOW_MEMORY", "").lower() in ("1", "true", "yes") or os.environ.get(
        "FORCE_CPU", ""
    ).lower() in ("1", "true", "yes")


def configure_torch():
    """Call once at process startup before loading the model."""
    global _torch_configured
    if _torch_configured:
        return

    try:
        import torch
    except ImportError:
        return

    threads = int(os.environ.get("TORCH_NUM_THREADS", "1" if is_low_memory_mode() else "2"))
    try:
        torch.set_num_threads(max(1, threads))
    except RuntimeError:
        pass

    torch.set_grad_enabled(False)

    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already set or parallel work has started

    _torch_configured = True


def release_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
