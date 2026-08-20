"""Full default stack on CUDA against a shipped mitten code."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ortools")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from spyglass import TelescopingDecoder
from spyglass.noise import sample_code_capacity

MITTEN_150 = (Path(__file__).resolve().parents[2] / "processor_codes"
              / "mitten" / "[[150,30,10]]")


def test_mitten_150_full_stack():
    if not MITTEN_150.exists():
        pytest.skip("processor_codes not present")
    H = np.load(MITTEN_150 / "Hx.npy").astype(np.uint8)
    dec = TelescopingDecoder(H, np.full(H.shape[1], 0.03), device="cuda",
                             seed=9)
    _, syndromes = sample_code_capacity(H, 0.03, 2000,
                                        np.random.default_rng(23))
    result = dec.decode_batch(syndromes)
    assert result.converged.all()
    assert np.array_equal(result.corrections @ H.T % 2, syndromes)
    shots_in = result.telemetry.shots_in
    assert all(a > b for a, b in zip(shots_in, shots_in[1:])), shots_in
