#!/usr/bin/env bash
# Run with bash from a writable checkout; no notebook or activation is required.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_VENV="${KAGGLE_VENV:-/tmp/robomimic-kaggle-venv}"
KAGGLE_PYTHON="${KAGGLE_PYTHON:-python3}"
# Keep the base interpreter after PATH starts preferring the new venv.
KAGGLE_PYTHON="$("$KAGGLE_PYTHON" -c 'import sys; print(sys.executable)')"

"$KAGGLE_PYTHON" - <<'PY'
import platform
import sys

if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("This installer targets Kaggle's Linux x86_64 GPU runtime.")
if not (3, 12) <= sys.version_info[:2] <= (3, 13):
    raise SystemExit(
        "Use CPython 3.12 or 3.13 (current Kaggle image: 3.13). "
        "Set KAGGLE_PYTHON to that interpreter if necessary."
    )
PY

# Keep the multi-GB environment out of /kaggle/working notebook outputs.
# Re-running the installer updates this venv; it never changes the base image.
# Some hosted Python images cannot run ensurepip. Create the environment without
# it, then use the base image's pip to install pip into this specific venv.
# This also repairs an environment left partially created by the old installer.
"$KAGGLE_PYTHON" -m venv --without-pip "$KAGGLE_VENV"
export PATH="$KAGGLE_VENV/bin:$PATH"
export PIP_REQUIRE_VIRTUALENV=true
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
unset PYTHONPATH PYTHONHOME

# egl_probe uses a legacy CMake project. Install this before pip builds it.
"$KAGGLE_PYTHON" -m pip --python "$KAGGLE_VENV/bin/python" install --upgrade \
    'pip>=25.1' 'setuptools<82' wheel 'cmake<4'
# Build this package in the venv so its CMake launcher can import the cmake
# Python package; pip's isolated build environment would hide that package.
"$KAGGLE_VENV/bin/python" -m pip install --no-build-isolation 'egl_probe==1.0.2'
"$KAGGLE_VENV/bin/python" -m pip install \
    -r "$REPO_DIR/requirements-kaggle-t4.txt" -e "$REPO_DIR"
"$KAGGLE_VENV/bin/python" -m pip check

# CUDA is required here. Run check_kaggle_t4.py without --skip-render to check EGL.
"$KAGGLE_VENV/bin/python" "$REPO_DIR/examples/check_kaggle_t4.py" --skip-render
