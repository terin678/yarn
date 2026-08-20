"""CUDA behavior: CPU parity, batch invariance, bitwise determinism.

Determinism across runs is a design guarantee of the padded-dense layout
(no scatter atomics); this file is where a regression would surface."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from spyglass import build_problem
from spyglass.bp import bp_decode_batch
from spyglass.noise import sample_code_capacity

MITTEN_150 = (Path(__file__).resolve().parents[2] / "processor_codes"
              / "mitten" / "[[150,30,10]]")


def _problems():
    steane = np.array([[0, 0, 0, 1, 1, 1, 1],
                       [0, 1, 1, 0, 0, 1, 1],
                       [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    out = [("steane", steane)]
    if MITTEN_150.exists():
        out.append(("mitten150", np.load(MITTEN_150 / "Hx.npy")
                    .astype(np.uint8)))
    return out


@pytest.mark.parametrize("name,H", _problems())
def test_cuda_matches_cpu_hard_decisions(name, H):
    problem = build_problem(H, np.full(H.shape[1], 0.01))
    _, syndromes = sample_code_capacity(H, 0.01, 1000,
                                        np.random.default_rng(2))
    r_cpu = bp_decode_batch(problem, syndromes, max_iter=60, device="cpu")
    r_gpu = bp_decode_batch(problem, syndromes, max_iter=60, device="cuda")
    assert np.array_equal(r_cpu.converged, r_gpu.converged), name
    assert np.array_equal(r_cpu.hard, r_gpu.hard), name
    assert np.allclose(r_cpu.posteriors, r_gpu.posteriors, atol=1e-4), name


def test_batch_invariance_on_cuda():
    _, H = _problems()[-1]
    problem = build_problem(H, np.full(H.shape[1], 0.01))
    _, syndromes = sample_code_capacity(H, 0.01, 256,
                                        np.random.default_rng(9))
    r_all = bp_decode_batch(problem, syndromes, max_iter=40, device="cuda")
    for i in (0, 17, 100, 255):
        r_one = bp_decode_batch(problem, syndromes[i][None], max_iter=40,
                                device="cuda")
        assert np.array_equal(r_all.hard[i], r_one.hard[0]), i
        assert r_all.converged[i] == r_one.converged[0], i


def test_bitwise_determinism_across_runs():
    _, H = _problems()[-1]
    problem = build_problem(H, np.full(H.shape[1], 0.02))
    _, syndromes = sample_code_capacity(H, 0.02, 512,
                                        np.random.default_rng(4))
    r1 = bp_decode_batch(problem, syndromes, max_iter=60, device="cuda",
                         return_state=True)
    r2 = bp_decode_batch(problem, syndromes, max_iter=60, device="cuda",
                         return_state=True)
    assert np.array_equal(r1.hard, r2.hard)
    assert np.array_equal(r1.iterations, r2.iterations)
    assert np.array_equal(r1.state_c2v, r2.state_c2v)  # bitwise
