"""Pipeline driver bookkeeping, refereed by scripted mock stages.

The driver's job is residual index mapping, stage-of-resolution tracking,
posterior threading, telemetry, and graceful degradation; the mocks make
each of those observable without any real decoder."""

import numpy as np
import pytest

from spyglass import build_problem
from spyglass.pipeline import TelescopingDecoder
from spyglass.stages import StageOutput

pytestmark = pytest.mark.fast

H = np.array([[1, 1, 0],
              [0, 1, 1]], dtype=np.uint8)


class ScriptedStage:
    """Resolves the shots whose ORIGINAL batch index is in `resolve`;
    stamps corrections with `tag` so provenance is visible."""

    def __init__(self, name, resolve, tag, posterior_note=None,
                 raise_error=False):
        self.name = name
        self.resolve = set(resolve)
        self.tag = tag
        self.posterior_note = posterior_note
        self.raise_error = raise_error
        self.seen_posteriors = None
        self.seen_batch = None

    def decode_batch(self, problem, syndromes, posteriors_in):
        if self.raise_error:
            raise RuntimeError("scripted failure")
        self.seen_posteriors = posteriors_in
        self.seen_batch = syndromes.shape[0]
        B = syndromes.shape[0]
        n = problem.H.shape[1]
        corrections = np.full((B, n), self.tag, dtype=np.uint8)
        # the driver passes original indices for observability
        resolved = np.array([idx in self.resolve
                             for idx in self._original_idx], dtype=bool)
        posteriors = (np.full((B, n), self.posterior_note)
                      if self.posterior_note is not None else None)
        return StageOutput(corrections=corrections, resolved=resolved,
                           posteriors=posteriors, stats={})


def _make(problem, stages):
    dec = TelescopingDecoder(problem.H, problem.priors, stages=stages)
    return dec


@pytest.fixture()
def problem():
    return build_problem(H, np.full(3, 0.1))


def _wire_original_idx(stages, decoder):
    # ScriptedStage needs to know which original shots it is looking at;
    # the driver exposes the residual mapping on itself for telemetry, and
    # the test hooks it per call
    class Wrapper:
        def __init__(self, inner, decoder):
            self.inner = inner
            self.name = inner.name
            self.decoder = decoder

        def decode_batch(self, problem, syndromes, posteriors_in):
            self.inner._original_idx = self.decoder.current_residual_idx
            return self.inner.decode_batch(problem, syndromes, posteriors_in)

    return [Wrapper(s, decoder) for s in stages]


def test_stage_of_resolution_and_residual_mapping(problem):
    s0 = ScriptedStage("a", resolve={0, 3}, tag=1)
    s1 = ScriptedStage("b", resolve={1, 4}, tag=2)
    dec = TelescopingDecoder(problem.H, problem.priors, stages=[])
    dec.stages = _wire_original_idx([s0, s1], dec)
    syndromes = np.zeros((5, 2), dtype=np.uint8)
    result = dec.decode_batch(syndromes)
    assert result.stage.tolist() == [0, 1, -1, 0, 1]
    # stage 1 saw only the residual of stage 0 (shots 1, 2, 4)
    assert s1.seen_batch == 3
    # resolved corrections carry their resolving stage's tag
    assert result.corrections[0, 0] == 1 and result.corrections[3, 0] == 1
    assert result.corrections[1, 0] == 2 and result.corrections[4, 0] == 2
    # unresolved shot keeps the LAST stage's best-effort correction
    assert result.corrections[2, 0] == 2
    assert not result.converged[2]
    assert result.converged[[0, 1, 3, 4]].all()


def test_posteriors_thread_between_stages(problem):
    s0 = ScriptedStage("a", resolve={0}, tag=1, posterior_note=0.25)
    s1 = ScriptedStage("b", resolve={1, 2}, tag=2)
    dec = TelescopingDecoder(problem.H, problem.priors, stages=[])
    dec.stages = _wire_original_idx([s0, s1], dec)
    dec.decode_batch(np.zeros((3, 2), dtype=np.uint8))
    assert s1.seen_posteriors is not None
    assert np.allclose(s1.seen_posteriors, 0.25)


def test_telemetry_counts(problem):
    s0 = ScriptedStage("a", resolve={0, 3}, tag=1)
    s1 = ScriptedStage("b", resolve={1, 4}, tag=2)
    dec = TelescopingDecoder(problem.H, problem.priors, stages=[])
    dec.stages = _wire_original_idx([s0, s1], dec)
    result = dec.decode_batch(np.zeros((5, 2), dtype=np.uint8))
    t = result.telemetry
    assert t.stage_names == ("a", "b")
    assert t.shots_in == [5, 3]
    assert t.shots_resolved == [2, 2]


def test_raising_stage_is_skipped_not_fatal(problem):
    s0 = ScriptedStage("a", resolve=set(), tag=1, raise_error=True)
    s1 = ScriptedStage("b", resolve={0, 1}, tag=2)
    dec = TelescopingDecoder(problem.H, problem.priors, stages=[])
    dec.stages = _wire_original_idx([s0, s1], dec)
    result = dec.decode_batch(np.zeros((2, 2), dtype=np.uint8))
    assert result.stage.tolist() == [1, 1]
    assert "scripted failure" in result.telemetry.stage_errors["a"]


def test_early_exit_when_all_resolved(problem):
    s0 = ScriptedStage("a", resolve={0, 1}, tag=1)
    s1 = ScriptedStage("b", resolve=set(), tag=2)
    dec = TelescopingDecoder(problem.H, problem.priors, stages=[])
    dec.stages = _wire_original_idx([s0, s1], dec)
    result = dec.decode_batch(np.zeros((2, 2), dtype=np.uint8))
    assert result.stage.tolist() == [0, 0]
    assert result.telemetry.shots_in == [2]  # stage b never ran


def test_single_shot_decode_wraps_batch(problem):
    s0 = ScriptedStage("a", resolve={0}, tag=7)
    dec = TelescopingDecoder(problem.H, problem.priors, stages=[])
    dec.stages = _wire_original_idx([s0], dec)
    corr = dec.decode(np.zeros(2, dtype=np.uint8))
    assert corr.shape == (3,) and corr[0] == 7
