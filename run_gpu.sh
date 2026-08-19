#!/bin/bash
# JIT env for sqetch on native Windows: full vcvars64 environment (captured
# in .vcenv) + CUDA 13.3 + venv Scripts, msys-style PATH entries.
cd "$(dirname "$0")"
while IFS='=' read -r k v; do
  case "$k" in
    INCLUDE|LIB|LIBPATH) export "$k=$v" ;;
  esac
done < .vcenv
MSVC="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
CUDA_MSYS="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
export PATH="$(pwd)/.venv/Scripts:$MSVC:$CUDA_MSYS/bin:$PATH"
export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
export CUDA_PATH="$CUDA_HOME"
export DISTUTILS_USE_SDK=1
exec ./.venv/Scripts/python.exe "$@"
