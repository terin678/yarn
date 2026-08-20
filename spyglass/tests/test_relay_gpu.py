"""Relay on CUDA: it must rescue shots plain BP leaves behind, and stay
seed-deterministic on the device."""

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
from spyglass.relay import RelayBpStage

MITTEN_150 = (Path(__file__).resolve().parents[2] / "processor_codes"
              / "mitten" / "[[150,30,10]]")


@pytest.fixture(scope="module")
def mitten():
    if not MITTEN_150.exists():
        pytest.skip("processor_codes not present")
    return np.load(MITTEN_150 / "Hx.npy").astype(np.uint8)


def test_relay_rescues_shots_plain_bp_leaves(mitten):
    p = 0.05  # hard enough that plain BP leaves a residual
    problem = build_problem(mitten, np.full(mitten.shape[1], p))
    _, syndromes = sample_code_capacity(mitten, p, 2000,
                                        np.random.default_rng(12))
    plain = bp_decode_batch(problem, syndromes, max_iter=240, device="cuda")
    residual = syndromes[~plain.converged]
    assert residual.shape[0] > 0, "raise p; plain BP resolved everything"

    stage = RelayBpStage(num_legs=12, leg_max_iter=20, seed=7,
                         device="cuda")
    out = stage.decode_batch(problem, residual,
                             plain.posteriors[~plain.converged])
    assert out.resolved.sum() > 0
    ok = out.corrections[out.resolved] @ problem.H.T % 2
    assert np.array_equal(ok, residual[out.resolved])


def test_relay_cuda_seed_determinism(mitten):
    problem = build_problem(mitten, np.full(mitten.shape[1], 0.04))
    _, syndromes = sample_code_capacity(mitten, 0.04, 512,
                                        np.random.default_rng(13))
    r1 = RelayBpStage(num_legs=6, leg_max_iter=20, seed=11, device="cuda")\
        .decode_batch(problem, syndromes, None)
    r2 = RelayBpStage(num_legs=6, leg_max_iter=20, seed=11, device="cuda")\
        .decode_batch(problem, syndromes, None)
    assert np.array_equal(r1.resolved, r2.resolved)
    assert np.array_equal(r1.corrections, r2.corrections)
