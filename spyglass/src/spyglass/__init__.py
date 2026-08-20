"""spyglass: a staged predecode-and-escalate decoder for quantum LDPC codes.

Unofficial implementation of the telescoping decoder architecture
described in arXiv 2607.28795 Appendix I.A. All staging constants and
kernels are this implementation's own.
"""

from ._problem import DecodingProblem, build_problem
from .noise import code_capacity_problem, sample_code_capacity
from .pipeline import DecodeResult, TelescopingDecoder, default_stages
from .stages import Stage, StageOutput, StageTelemetry

__all__ = [
    "DecodingProblem",
    "build_problem",
    "code_capacity_problem",
    "sample_code_capacity",
    "DecodeResult",
    "TelescopingDecoder",
    "default_stages",
    "Stage",
    "StageOutput",
    "StageTelemetry",
]
