# Running sqetch on native Windows

Verified configuration: Windows 11, RTX 4090, MSVC 14.44 (VS 2022), CUDA
toolkit 13.3, Python 3.11 venv, torch 2.13.0+cu130. Steane quickstart
returns d_Z = 3 at ~15.8k trials/s; the full test suite passes (9 passed,
3 skipped for absent second GPU).

## Setup steps that differ from the README

1. **torch**: PyPI's default Windows torch wheel is CPU-only (the README's
   platform note says otherwise). Install a CUDA build explicitly:

   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   ```

2. **ninja**: required by torch's extension JIT and not declared by the
   `[gpu]` extra. `pip install ninja`, and make sure the venv's Scripts
   directory is on PATH when python runs.

3. **CUDA toolkit**: a full toolkit install matching torch's CUDA major
   version is required for the JIT (the kernel compiles at first import).
   NVIDIA's pip `cuda-nvcc` wheel for Windows ships `ptxas` but no `nvcc`,
   so pip alone cannot satisfy this. `winget install Nvidia.CUDA` works.

4. **MSVC environment**: torch's JIT invokes `cl` and needs the full
   vcvars64 environment (INCLUDE/LIB), not just cl.exe on PATH. See
   `run_gpu.sh` at the repo root for a wrapper that captures and replays
   it from git-bash.

## Code changes in this fork

- `api.py` passes `-Xcompiler /Zc:preprocessor` to nvcc on Windows: CUDA
  12.4+ cccl headers hard-error under MSVC's traditional preprocessor.
- The JIT build directory comes from `tempfile.gettempdir()` rather than
  a hardcoded `/tmp`.
