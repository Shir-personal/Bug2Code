"""Reproducibility helpers: seeding and device selection."""

from __future__ import annotations

import os
import random

import numpy as np

from bug2code.logging_utils import get_logger

logger = get_logger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch on all devices."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; seeded python and numpy only")
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string, mapping ``auto`` to the best available backend."""
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
