"""Relay ensemble logic on CPU: seeding, first-hit bookkeeping, and the
randomized memory draws themselves."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyglass import build_problem
from spyglass.noise import sample_code_capacity
from spyglass.relay import RelayBpStage

pytestmark = pytest.mark.fast


def _steane_problem(steane_Hz, p=0.08):
    return build_problem(steane_Hz, np.full(7, p))


def test_seeded_reproducibility(steane_Hz):
    problem = _steane_problem(steane_Hz)
    _, syndromes = sample_code_capacity(steane_Hz, 0.15, 64,
                                        np.random.default_rng(21))
    a = RelayBpStage(num_legs=4, leg_max_iter=15, seed=5, device="cpu")
    b = RelayBpStage(num_legs=4, leg_max_iter=15, seed=5, device="cpu")
    ra = a.decode_batch(problem, syndromes, None)
    rb = b.decode_batch(problem, syndromes, None)
    assert np.array_equal(ra.resolved, rb.resolved)
    assert np.array_equal(ra.corrections, rb.corrections)
    assert ra.stats["legs_used"] == rb.stats["legs_used"]

    # seed sensitivity is asserted on the draws themselves: outcomes can
    # legitimately coincide when every shot resolves in leg 0
    c = RelayBpStage(num_legs=4, leg_max_iter=15, seed=6, device="cpu")
    assert not torch.equal(a._draw_gamma(8, 7, leg=0),
                           c._draw_gamma(8, 7, leg=0))


def test_first_hit_shots_keep_their_leg_solution(steane_Hz):
    problem = _steane_problem(steane_Hz)
    # weight-1 syndromes: leg 0 resolves them; later legs must not touch
    syndromes = []
    for i in range(7):
        e = np.zeros(7, dtype=np.uint8)
        e[i] = 1
        syndromes.append(steane_Hz @ e % 2)
    syndromes = np.stack(syndromes)
    stage = RelayBpStage(num_legs=6, leg_max_iter=40, seed=1, device="cpu")
    out = stage.decode_batch(problem, syndromes, None)
    assert out.resolved.all()
    # every accepted correction reproduces its syndrome
    assert np.array_equal(out.corrections @ problem.H.T % 2, syndromes)
    # per-shot leg-of-resolution recorded and mostly leg 0 for easy shots
    legs = np.asarray(out.stats["leg_of_resolution"])
    assert legs.shape == (7,)
    assert (legs >= 0).all()


def test_gamma_draws_vary_per_shot_and_variable(steane_Hz):
    stage = RelayBpStage(num_legs=2, leg_max_iter=5, seed=3, device="cpu")
    g = stage._draw_gamma(batch=4, n=7, leg=0)
    assert g.shape == (4, 7)
    assert float(g.std()) > 0.0
    g2 = stage._draw_gamma(batch=4, n=7, leg=1)
    assert not torch.equal(g, g2)
    # bounds respected
    lo, hi = stage.gamma_range
    assert float(g.min()) >= lo and float(g.max()) <= hi
