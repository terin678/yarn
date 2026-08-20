"""Shared fixtures: tiny hand-built codes with known structure.

The repetition and Steane matrices are written out longhand so tests
referee the package against paper facts, not against itself.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def ring_repetition_H():
    """Length-5 ring repetition code: check i compares bits i and i+1."""
    H = np.zeros((5, 5), dtype=np.uint8)
    for i in range(5):
        H[i, i] = 1
        H[i, (i + 1) % 5] = 1
    return H


@pytest.fixture(scope="session")
def steane_Hz():
    """Hamming(7,4) parity check; Steane's Hz (and Hx)."""
    return np.array([[0, 0, 0, 1, 1, 1, 1],
                     [0, 1, 1, 0, 0, 1, 1],
                     [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)


@pytest.fixture(scope="session")
def steane_Lz():
    return np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
