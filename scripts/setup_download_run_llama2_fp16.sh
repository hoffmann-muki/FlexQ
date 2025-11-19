#!/usr/bin/env bash
set -euo pipefail

# This script prepares the conda `flexq` env, optionally downloads the
# Llama-2-7b HF snapshot to a local folder, installs helper packages,
# and runs an example evaluation using the repo's Python evaluation script.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="flexq"
MODEL_DIR="${REPO_ROOT}/models/llama-2-7b-hf"
HF_REPO="meta-llama/Llama-2-7b-hf"

echo "Repo root: ${REPO_ROOT}"

# Ensure conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Please install Anaconda/Miniconda and retry."
  exit 1
fi

# Source conda.sh to enable 'conda activate' in scripts
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "WARNING: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh"
fi

# Create the conda env if it doesn't exist
if conda env list | awk '{print $1}' | grep -xq "$CONDA_ENV"; then
  echo "Conda env '$CONDA_ENV' already exists. Activating it."
else
  echo "Creating conda env '$CONDA_ENV' with python=3.10"
  conda create -n "$CONDA_ENV" python=3.10 -y
fi

echo "Activating conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

echo "Upgrading pip/setuptools/wheel and installing helper packages"
python -m pip install --upgrade pip setuptools wheel
python -m pip install huggingface_hub safetensors accelerate

# Check for a Hugging Face token
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "No HUGGINGFACE_HUB_TOKEN environment variable detected. You will need to login interactively."
  echo "If you prefer non-interactive, set HUGGINGFACE_HUB_TOKEN in the environment before running this script."
  read -p "Do you want to run 'huggingface-cli login' now? [y/N]: " do_login
  if [[ "${do_login}" =~ ^[Yy]$ ]]; then
    huggingface-cli login
  else
    echo "Continuing without interactive login. Downloads from private/consented repos may fail."
  fi
else
  echo "Using HUGGINGFACE_HUB_TOKEN from environment."
fi

# Offer to download the model snapshot locally
read -p "Download HF snapshot to ${MODEL_DIR}? [y/N]: " do_download
if [[ "${do_download}" =~ ^[Yy]$ ]]; then
  echo "Downloading snapshot of ${HF_REPO} into ${MODEL_DIR} (can be large)"
  python - <<PY
from huggingface_hub import snapshot_download
print('Starting snapshot_download...')
snapshot_download(repo_id='${HF_REPO}', cache_dir='${MODEL_DIR}')
print('snapshot_download finished')
PY
  echo "Model snapshot downloaded to ${MODEL_DIR}"
else
  echo "Skipping local download. The evaluation script will download on demand from Hugging Face."
fi

# Run example evaluation
read -p "Run example accuracy evaluation now (may take long)? [y/N]: " do_run
if [[ "${do_run}" =~ ^[Yy]$ ]]; then
  # If the local snapshot exists, use it; otherwise use the HF repo id
  if [ -d "${MODEL_DIR}" ]; then
    MODEL_ARG="${MODEL_DIR}"
  else
    MODEL_ARG="${HF_REPO}"
  fi
  echo "Running evaluation using model: ${MODEL_ARG}"
  # Example: evaluate perplexity and a set of tasks. Adjust as needed.
  python "${REPO_ROOT}/algorithm/main.py" \
    --model "${MODEL_ARG}" \
    --net Llama-2-7b \
    --eval_ppl \
    --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande \
    --deactive_amp
else
  echo "Done. You can run the evaluation later with the command printed below."
  echo
  echo "Example run command (local snapshot):"
  echo "python ${REPO_ROOT}/algorithm/main.py --model ${MODEL_DIR} --net Llama-2-7b --eval_ppl --deactive_amp"
  echo
  echo "Example run command (download on demand):"
  echo "python ${REPO_ROOT}/algorithm/main.py --model ${HF_REPO} --net Llama-2-7b --eval_ppl --deactive_amp"
fi

echo "Script finished."
