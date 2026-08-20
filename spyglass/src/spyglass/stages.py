"""The stage seam: what every pipeline stage produces and promises.

A stage receives the problem, a batch of syndromes (the current
residual), and optional soft information from the previous stage. It must
return a row for EVERY input shot: ``resolved`` marks the shots whose
hard decision reproduces the syndrome exactly (the escalation criterion
throughout the pipeline), and ``corrections`` rows for unresolved shots
should carry the stage's best effort, because the driver keeps the last
stage's attempt for shots nothing resolves.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from ._problem import DecodingProblem


@dataclass
class StageOutput:
    corrections: np.ndarray            # (B_stage, n) uint8
    resolved: np.ndarray               # (B_stage,) bool
    posteriors: Optional[np.ndarray]   # (B_stage, n) soft info, or None
    stats: dict = field(default_factory=dict)


class Stage(Protocol):
    name: str

    def decode_batch(self, problem: DecodingProblem, syndromes: np.ndarray,
                     posteriors_in: Optional[np.ndarray]) -> StageOutput:
        ...


@dataclass
class StageTelemetry:
    """Per-stage aggregate counters for the stages that actually ran."""

    stage_names: tuple = ()
    shots_in: list = field(default_factory=list)
    shots_resolved: list = field(default_factory=list)
    wall_seconds: list = field(default_factory=list)
    stage_stats: list = field(default_factory=list)
    stage_errors: dict = field(default_factory=dict)  # name -> message
