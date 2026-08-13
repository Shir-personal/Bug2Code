import random

import numpy as np

from bug2code.seed import resolve_device, set_seed


def test_set_seed_makes_python_and_numpy_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand())
    set_seed(123)
    assert (random.random(), np.random.rand()) == first


def test_resolve_device_passes_through_explicit_choice():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_known_backend():
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
