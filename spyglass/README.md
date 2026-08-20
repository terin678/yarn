# spyglass

A staged decoder for quantum LDPC codes: cheap batched belief propagation
handles the bulk of shots, and progressively more expensive stages act on
the shrinking residual of harder ones, ending in an exact most-likely-error
solve.

This is an **unofficial implementation of the telescoping decoder
architecture** described in arXiv 2607.28795 Appendix I.A. The staging
idea (nested BP and Relay-BP stages that each resolve the shots they can
and escalate the rest) follows the paper's description; **all staging
constants, kernels, and the exact endgame here are this implementation's
own** and are not claimed to reproduce the original decoder or its
reported numbers.

## Architecture

```
syndromes ->  [ batched BP (torch) ] -> residual
           -> [ Relay-BP ensemble  ] -> residual
           -> [ serial-schedule BP ] -> residual
           -> [ exact MLE (CP-SAT) ] -> corrections + telemetry
```

A shot is resolved the moment a stage's hard decision reproduces its
syndrome exactly; unresolved shots carry the last stage's best effort and
are flagged. Every stage records how many shots it saw and resolved, so a
decode returns its own cost profile.

## Scope

The decoder consumes a parity-check matrix and per-mechanism priors
(`DecodingProblem`). Code-capacity noise helpers are included; decoding
the two CSS directions is the caller's composition. **Circuit-level
detector error models are out of scope here** because the upstream
release does not ship measurement schedules; the interface is
deliberately DEM-ready (a detector error model is just a different
`(H, priors)` pair), so nothing structural changes when one exists.

## Install

```
pip install -e .            # numpy core
pip install -e .[gpu]       # + torch batched BP stages
pip install -e .[serialbp]  # + ldpc serial-schedule stage
pip install -e .[endgame]   # + OR-Tools exact MLE stage
```

Missing optional stages are skipped with a note in telemetry; the
pipeline never crashes because a backend is absent.

## Usage

```python
import numpy as np
from spyglass import TelescopingDecoder, code_capacity_problem
from spyglass.noise import sample_code_capacity

Hx = np.load(".../processor_codes/mitten/[[150,30,10]]/Hx.npy")
errors, syndromes = sample_code_capacity(Hx, 0.01, shots=10_000,
                                         rng=np.random.default_rng(1))
dec = TelescopingDecoder(Hx, np.full(Hx.shape[1], 0.01))
result = dec.decode_batch(syndromes)
# result.corrections, result.converged, result.stage, result.telemetry
```

## Tests

```
pytest tests -m fast    # numpy only
pytest tests -m gpu     # CUDA parity, batch invariance, determinism
pytest tests -m ldpc    # serial-schedule stage
```
