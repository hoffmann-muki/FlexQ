#!/usr/bin/env bash
set -euo pipefail

# Create a conda prefix env inside the repo and install project requirements
# Usage:
#   ./create_prefix_env_and_install.sh [--no-torch] [--torch-cuda cu121]
# Examples:
#   ./create_prefix_env_and_install.sh
#   TORCH_CUDA=cu121 ./create_prefix_env_and_install.sh
#   ./create_prefix_env_and_install.sh --no-torch

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX_PATH="${REPO_ROOT}/.conda-env"
REQ_FILE="${REPO_ROOT}/algorithm/requirements.txt"
NO_TORCH=0
TORCH_CUDA="${TORCH_CUDA:-cu121}"

while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --no-torch)
      NO_TORCH=1
      shift
      ;;
    --torch-cuda)
      TORCH_CUDA="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,200p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"; exit 1
      ;;
  esac
done

echo "Repo root: ${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Install Anaconda/Miniconda and retry." >&2
  exit 1
fi

CONDA_BASE=$(conda info --base)
# Source conda.sh for activation support
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# Create prefix env if not exists
if [ -d "$PREFIX_PATH" ]; then
  echo "Prefix env already exists at $PREFIX_PATH"
else
  echo "Creating prefix env at $PREFIX_PATH (python=3.10)"
  conda create --prefix "$PREFIX_PATH" python=3.10 -y
fi

# Use conda run to execute pip inside the prefix env (avoids activation in scripts)
echo "Upgrading pip and installing build tools in env"
conda run --prefix "$PREFIX_PATH" python -m pip install --upgrade pip setuptools wheel

# Install requirements (excluding torch) -- build a temporary requirements file
TMP_REQ="/tmp/flexq_requirements_no_torch.txt"
grep -vi '^torch' "$REQ_FILE" > "$TMP_REQ" || true

echo "Installing Python requirements from $REQ_FILE (excluding torch) into prefix env"
conda run --prefix "$PREFIX_PATH" python -m pip install -r "$TMP_REQ"

# Install torch optionally (choose CUDA build via TORCH_CUDA env or default)
if [ "$NO_TORCH" -eq 1 ]; then
  echo "Skipping torch install (--no-torch)"
else
  echo "Installing torch==2.2.0 using index for ${TORCH_CUDA} (if available)"
  set +e
  conda run --prefix "$PREFIX_PATH" python -m pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA}/ torch==2.2.0
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "Failed to install from cu wheel index; falling back to plain pip install torch==2.2.0"
    conda run --prefix "$PREFIX_PATH" python -m pip install torch==2.2.0
  fi
  set -e
fi

# Quick smoke test
echo "Running smoke checks inside prefix env"
conda run --prefix "$PREFIX_PATH" python - <<'PY'
import sys
import importlib
print('python', sys.version.split()[0])
for pkg in ['torch','transformers','datasets','accelerate','numpy']:
    try:
        m=importlib.import_module(pkg)
        print(pkg, getattr(m,'__version__',None))
    except Exception as e:
        print(pkg, 'import error:', e)
PY

echo "Prefix env creation and package installation complete. Env path: $PREFIX_PATH"
echo "To use it interactively, run:"
echo "  conda activate $PREFIX_PATH"
echo "Or use conda run --prefix $PREFIX_PATH <command>"
