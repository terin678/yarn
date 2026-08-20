"""The telescoping driver: run stages over a shrinking residual.

This package is an unofficial implementation of the telescoping decoder
architecture described in arXiv 2607.28795 Appendix I.A: nested stages of
progressively more expensive decoders, each resolving the shots it can
and passing the residual onward. All staging constants and kernels here
are this implementation's own.

The driver owns the bookkeeping and nothing else: residual index
mapping, per-shot stage-of-resolution, soft-information threading, and
telemetry. A stage that raises is recorded and skipped; shots nothing
resolves come back with ``stage == -1`` and the last stage's best-effort
correction. The driver never raises on decoding outcomes.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._problem import DecodingProblem, build_problem
from .stages import Stage, StageTelemetry


@dataclass(frozen=True)
class DecodeResult:
    corrections: np.ndarray      # (B, n) uint8
    converged: np.ndarray        # (B,) bool: syndrome reproduced exactly
    stage: np.ndarray            # (B,) int8 index into stage_names; -1 = none
    stage_names: tuple
    telemetry: StageTelemetry


def default_stages(*, device: str = "auto", seed: Optional[int] = None,
                   endgame_time_limit_s: float = 10.0) -> list:
    """The standard stack: batched BP, the relay ensemble, then the
    optional serial-BP and exact-endgame backstops when their backends
    are importable. ``device="auto"`` picks CUDA when available."""
    from .bp import BatchBpStage
    from .relay import RelayBpStage

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    stages: list = [
        BatchBpStage(max_iter=200, device=device),
        RelayBpStage(num_legs=12, leg_max_iter=30, seed=seed,
                     device=device),
    ]
    try:
        from .cpu_stage import SerialBpStage
        import ldpc  # noqa: F401
        stages.append(SerialBpStage(max_iter=200))
    except ImportError:
        pass
    try:
        from .endgame import CpSatEndgame
        from ortools.sat.python import cp_model  # noqa: F401
        stages.append(CpSatEndgame(time_limit_s=endgame_time_limit_s))
    except ImportError:
        pass
    return stages


class TelescopingDecoder:
    """Staged decoder over a fixed ``(H, priors)`` problem.

    Args:
        H: parity-check matrix, ``(m, n)`` binary.
        priors: per-mechanism flip probabilities, ``(n,)`` in (0, 0.5).
        stages: ordered stage list; defaults to :func:`default_stages`.
    """

    def __init__(self, H: np.ndarray, priors: np.ndarray, *,
                 stages: Optional[list] = None, device: str = "auto",
                 seed: Optional[int] = None):
        self.problem: DecodingProblem = build_problem(H, priors)
        self.stages: list[Stage] = (stages if stages is not None
                                    else default_stages(device=device,
                                                        seed=seed))
        # exposed for telemetry and tests: original-batch indices of the
        # shots the currently running stage is seeing
        self.current_residual_idx: np.ndarray = np.empty(0, dtype=np.int64)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Single-shot form of :meth:`decode_batch`; the rig protocol."""
        return self.decode_batch(np.asarray(syndrome)[None])\
            .corrections[0]

    def decode_batch(self, syndromes: np.ndarray) -> DecodeResult:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        B = syndromes.shape[0]
        n = self.problem.H.shape[1]

        corrections = np.zeros((B, n), dtype=np.uint8)
        stage_of = np.full(B, -1, dtype=np.int8)
        residual = np.arange(B, dtype=np.int64)
        posteriors: Optional[np.ndarray] = None
        telemetry = StageTelemetry()
        names: list[str] = []

        for stage_index, stage in enumerate(self.stages):
            if residual.size == 0:
                break
            self.current_residual_idx = residual
            t0 = time.perf_counter()
            try:
                out = stage.decode_batch(self.problem, syndromes[residual],
                                         posteriors)
            except Exception as exc:  # a broken stage must not kill shots
                telemetry.stage_errors[stage.name] = repr(exc)
                continue
            wall = time.perf_counter() - t0

            names.append(stage.name)
            telemetry.shots_in.append(int(residual.size))
            telemetry.shots_resolved.append(int(out.resolved.sum()))
            telemetry.wall_seconds.append(round(wall, 6))
            telemetry.stage_stats.append(dict(out.stats))

            # every input shot gets this stage's attempt; resolved shots
            # keep it forever, unresolved shots keep it until a later
            # stage overwrites it
            corrections[residual] = out.corrections.astype(np.uint8)
            resolved_global = residual[out.resolved]
            stage_of[resolved_global] = stage_index

            residual = residual[~out.resolved]
            posteriors = (out.posteriors[~out.resolved]
                          if out.posteriors is not None else None)

        telemetry.stage_names = tuple(names)
        return DecodeResult(
            corrections=corrections,
            converged=stage_of >= 0,
            stage=stage_of,
            stage_names=tuple(s.name for s in self.stages),
            telemetry=telemetry)
