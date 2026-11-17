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
# Kaggle exports a Jupyter inline backend that is absent from the isolated venv.
export MPLBACKEND=Agg
export PIP_REQUIRE_VIRTUALENV=true
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
unset PYTHONPATH PYTHONHOME

# egl_probe uses a legacy CMake project. Install this before pip builds it.
"$KAGGLE_PYTHON" -m pip --python "$KAGGLE_VENV/bin/python" install --upgrade \
    'pip>=25.1' 'setuptools<82' wheel 'cmake<4' packaging 'wrapt>=1.17'

# Align the two legacy upstream pins before building EGL or downloading CUDA.
# Preserve a backup and stop on any other incompatible dependency declarations.
# Read setup.py without executing its build code.
"$KAGGLE_VENV/bin/python" - "$REPO_DIR" <<'PY'
import ast
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

repo = Path(sys.argv[1])
setup_path = repo / "setup.py"
setup_source = setup_path.read_bytes().decode("utf-8")
tree = ast.parse(setup_source)
setup_call = next(
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
)
dependency_node = next(
    keyword.value for keyword in setup_call.keywords if keyword.arg == "install_requires"
)
dependencies = ast.literal_eval(dependency_node)
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

legacy_pins = {
    "huggingface-hub": ("0.23.4", "0.25.2"),
    "transformers": ("4.41.2", "4.48.3"),
}
repairs = []
conflicts = []
for dependency, value_node in zip(dependencies, dependency_node.elts):
    requirement = Requirement(dependency)
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    name = canonicalize_name(requirement.name)
    pin = pins.get(name)
    if pin is not None and not requirement.specifier.contains(pin, prereleases=True):
        legacy = legacy_pins.get(name)
        if (
            legacy is not None and pin == legacy[1]
            and str(requirement.specifier) == f"=={legacy[0]}"
            and not requirement.extras and requirement.marker is None
            and dependency in ast.get_source_segment(setup_source, value_node)
        ):
            repairs.append((dependency, dependency.replace(legacy[0], pin, 1), value_node))
        else:
            conflicts.append(f"  setup.py: {dependency}; Kaggle requirements: {requirement.name}=={pin}")
if conflicts:
    raise SystemExit(
        "setup.py and requirements-kaggle-t4.txt have incompatible dependency pins:\n"
        + "\n".join(conflicts)
        + "\nCopy the updated setup.py from the modified checkout and rerun this script."
    )
if repairs:
    backup_path = repo / ".cache" / "kaggle" / "setup.py.before-kaggle"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        backup_path.write_bytes(setup_source.encode("utf-8"))
    source_bytes = setup_source.encode("utf-8")
    line_offsets = [0]
    for line in setup_source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line.encode("utf-8")))
    # AST columns are UTF-8 byte offsets. Edit only the dependency literals,
    # backwards through the file so other comments and source stay intact.
    for previous, updated, node in sorted(repairs, key=lambda item: (item[2].lineno, item[2].col_offset), reverse=True):
        start = line_offsets[node.lineno - 1] + node.col_offset
        end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        literal = source_bytes[start:end].decode("utf-8").replace(previous, updated, 1)
        source_bytes = source_bytes[:start] + literal.encode("utf-8") + source_bytes[end:]
        print(f"Updating setup.py: {previous} -> {updated}")
    setup_path.write_bytes(source_bytes)
    print(f"Original setup.py backup: {backup_path}")
print("Source dependency pins match the Kaggle requirements.")
PY

# Build this package in the venv so its CMake launcher can import the cmake
# Python package; pip's isolated build environment would hide that package.
"$KAGGLE_VENV/bin/python" -m pip install --no-build-isolation 'egl_probe==1.0.2'
"$KAGGLE_VENV/bin/python" -m pip install \
    -r "$REPO_DIR/requirements-kaggle-t4.txt" -e "$REPO_DIR"
"$KAGGLE_VENV/bin/python" -m pip check

# Check video-sized rendering before a long run reaches its first rollout.
# Training from stored images without rollouts can explicitly skip this check.
if [[ "${KAGGLE_SKIP_RENDER_CHECK:-0}" == "1" ]]; then
    "$KAGGLE_VENV/bin/python" "$REPO_DIR/examples/check_kaggle_t4.py" --skip-render
else
    "$KAGGLE_VENV/bin/python" "$REPO_DIR/examples/check_kaggle_t4.py"
fi
