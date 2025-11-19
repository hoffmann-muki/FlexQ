#!/usr/bin/env bash
set -euo pipefail

# Create a conda-pack archive for the project prefix env to allow relocation to similar hosts
# Usage:
#   ./create_condapack_archive.sh [--prefix ./flexq/.conda-env] [--output ./flexq/flexq-conda-env.tar.gz]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX_PATH="${1:-${REPO_ROOT}/.conda-env}"
OUTPUT_PATH="${2:-${REPO_ROOT}/flexq-conda-env.tar.gz}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Install Anaconda/Miniconda and retry." >&2
  exit 1
fi

echo "Prefix to pack: $PREFIX_PATH"
if [ ! -d "$PREFIX_PATH" ]; then
  echo "ERROR: prefix env path does not exist: $PREFIX_PATH" >&2
  exit 1
fi

# Ensure conda-pack is available. We'll try to run it; if missing, try to install into base env.
if ! python -c "import conda_pack" 2>/dev/null; then
  echo "conda-pack python module not found; attempting to install conda-pack into base conda installation"
  CONDA_BASE=$(conda info --base)
  if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  fi
  echo "Installing conda-pack into base"
  conda install -n base -c conda-forge conda-pack -y
fi

# Run conda-pack using the python -m conda_pack interface to avoid PATH issues
echo "Packing $PREFIX_PATH -> $OUTPUT_PATH (this can take a while)"
python -m conda_pack -p "$PREFIX_PATH" -o "$OUTPUT_PATH"

echo "Pack completed: $OUTPUT_PATH"

echo "To unpack on a target host (same OS/arch), extract and run conda-unpack inside the unpacked env":
echo "  tar -xzf ${OUTPUT_PATH} -C <target_dir>"
echo "  <target_dir>/bin/conda-unpack"
