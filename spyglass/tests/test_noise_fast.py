"""Code-capacity helpers: uniform priors, seeded sampling, syndrome
consistency."""

import numpy as np
import pytest

from spyglass import code_capacity_problem
from spyglass.noise import sample_code_capacity

pytestmark = pytest.mark.fast


def test_uniform_priors(steane_Hz):
    p = code_capacity_problem(steane_Hz, 0.05)
    assert np.allclose(p.priors, 0.05)


def test_sampling_is_seed_deterministic(steane_Hz):
    e1, s1 = sample_code_capacity(steane_Hz, 0.1, 200,
                                  np.random.default_rng(7))
    e2, s2 = sample_code_capacity(steane_Hz, 0.1, 200,
                                  np.random.default_rng(7))
    assert np.array_equal(e1, e2) and np.array_equal(s1, s2)
    e3, _ = sample_code_capacity(steane_Hz, 0.1, 200,
                                 np.random.default_rng(8))
    assert not np.array_equal(e1, e3)


def test_syndromes_match_errors(steane_Hz):
    errors, syndromes = sample_code_capacity(steane_Hz, 0.2, 50,
                                             np.random.default_rng(3))
    assert errors.shape == (50, 7) and syndromes.shape == (50, 3)
    assert np.array_equal(syndromes, errors @ steane_Hz.T % 2)
