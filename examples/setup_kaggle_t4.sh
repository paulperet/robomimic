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
    'pip>=25.1' 'setuptools<82' wheel 'cmake<4' packaging 'wrapt>=1.17'

# Catch mixed source / requirements revisions before building EGL or downloading
# the multi-GB CUDA stack. Read setup.py without executing its build code.
"$KAGGLE_VENV/bin/python" - "$REPO_DIR" <<'PY'
import ast
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

repo = Path(sys.argv[1])
tree = ast.parse((repo / "setup.py").read_text())
setup_call = next(
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
)
dependencies = ast.literal_eval(next(
    keyword.value for keyword in setup_call.keywords if keyword.arg == "install_requires"
))
pins = {}
for line in (repo / "requirements-kaggle-t4.txt").read_text().splitlines():
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        continue
    requirement = Requirement(line)
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    specifiers = list(requirement.specifier)
    if len(specifiers) == 1 and specifiers[0].operator == "==" and "*" not in specifiers[0].version:
        pins[canonicalize_name(requirement.name)] = specifiers[0].version

conflicts = []
for dependency in dependencies:
    requirement = Requirement(dependency)
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    pin = pins.get(canonicalize_name(requirement.name))
    if pin is not None and not requirement.specifier.contains(pin, prereleases=True):
        conflicts.append(f"  setup.py: {dependency}; Kaggle requirements: {requirement.name}=={pin}")
if conflicts:
    raise SystemExit(
        "setup.py and requirements-kaggle-t4.txt have incompatible dependency pins:\n"
        + "\n".join(conflicts)
        + "\nCopy the updated setup.py from the modified checkout and rerun this script."
    )
print("Source dependency pins match the Kaggle requirements.")
PY

# Build this package in the venv so its CMake launcher can import the cmake
# Python package; pip's isolated build environment would hide that package.
"$KAGGLE_VENV/bin/python" -m pip install --no-build-isolation 'egl_probe==1.0.2'
"$KAGGLE_VENV/bin/python" -m pip install \
    -r "$REPO_DIR/requirements-kaggle-t4.txt" -e "$REPO_DIR"
"$KAGGLE_VENV/bin/python" -m pip check

# CUDA is required here. Run check_kaggle_t4.py without --skip-render to check EGL.
"$KAGGLE_VENV/bin/python" "$REPO_DIR/examples/check_kaggle_t4.py" --skip-render
