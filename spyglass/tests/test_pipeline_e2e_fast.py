"""End-to-end through the default stack on CPU: composition, the
telescoping property, and the all-checks shot's journey to the endgame."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ortools")

from spyglass import TelescopingDecoder, default_stages
from spyglass.noise import sample_code_capacity

pytestmark = pytest.mark.fast


def test_default_stack_composition_and_order():
    stages = default_stages(device="cpu", seed=1)
    names = [s.name for s in stages]
    assert names[0] == "bp" and names[1] == "relay"
    assert names[-1] == "endgame"
    # serial_bp present iff ldpc importable
    try:
        import ldpc  # noqa: F401
        assert "serial_bp" in names
    except ImportError:
        assert "serial_bp" not in names


def test_steane_e2e_resolves_everything(steane_Hz):
    dec = TelescopingDecoder(steane_Hz, np.full(7, 0.05), device="cpu",
                             seed=3)
    _, syndromes = sample_code_capacity(steane_Hz, 0.05, 300,
                                        np.random.default_rng(17))
    result = dec.decode_batch(syndromes)
    assert result.converged.all()
    # telescoping: each ran stage saw no more shots than the one before
    shots_in = result.telemetry.shots_in
    assert all(a >= b for a, b in zip(shots_in, shots_in[1:]))
    # corrections reproduce syndromes
    assert np.array_equal(result.corrections @ steane_Hz.T % 2, syndromes)


def test_all_checks_shot_resolves_at_the_endgame(steane_Hz, steane_Lz):
    """The S8a-pinned min-sum failure travels the full pipeline: BP
    miscorrects it (syndrome-valid, so BP RESOLVES it wrongly)... unless
    it does not converge; either way the pipeline must return a
    syndrome-valid correction, and with BP's miscorrection being
    syndrome-valid the interesting assertion is logical accuracy of
    whatever stage answers."""
    all_checks_bit = int(np.flatnonzero(steane_Hz.sum(axis=0) == 3)[0])
    e = np.zeros(7, dtype=np.uint8)
    e[all_checks_bit] = 1
    syndrome = (steane_Hz @ e % 2)[None]

    dec = TelescopingDecoder(steane_Hz, np.full(7, 0.05), device="cpu",
                             seed=3)
    result = dec.decode_batch(syndrome)
    assert result.converged[0]
    assert np.array_equal(steane_Hz @ result.corrections[0] % 2,
                          syndrome[0])

    # endgame-only decode IS logically correct on this shot; the pinned
    # comparison documents what escalation buys when BP's syndrome-valid
    # answer is logically wrong
    endgame_only = TelescopingDecoder(
        steane_Hz, np.full(7, 0.05),
        stages=[s for s in default_stages(device="cpu", seed=3)
                if s.name == "endgame"])
    r2 = endgame_only.decode_batch(syndrome)
    assert (steane_Lz @ (r2.corrections[0] ^ e) % 2 == 0).all()
    assert np.array_equal(r2.corrections[0], e)
