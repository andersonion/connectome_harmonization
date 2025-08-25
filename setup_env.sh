#!/usr/bin/env bash


# setup_env.sh -- create/update conda env for this repo
# Puts env under ./.envs/neuroharmonize-env
# Uses conda-forge only (avoids Anaconda ToS)

set -euo pipefail

ENV_DIR="./.envs/neuroharmonize-env"
YML="environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Load module or install Miniconda/Mambaforge first." >&2
  exit 1
fi

if [[ ! -f "$YML" ]]; then
  echo "ERROR: $YML not found in current directory. Run from repo root." >&2
  exit 1
fi

mkdir -p "$(dirname "$ENV_DIR")"

# Use a temporary condarc to enforce conda-forge only
TMPCONDARC="$(mktemp)"
trap 'rm -f "$TMPCONDARC"' EXIT
cat >"$TMPCONDARC" <<'RC'
channels:
  - conda-forge
channel_priority: strict
RC
export CONDARC="$TMPCONDARC"

echo ">>> Conda version: $(conda --version)"
echo ">>> Creating/Updating env at $ENV_DIR"
echo ">>> Channels: conda-forge (strict priority)"

if conda env create -p "$ENV_DIR" -f "$YML"; then
  echo ">>> Environment created at $ENV_DIR"
else
  echo ">>> Environment may already exist; updating instead"
  conda env update -p "$ENV_DIR" -f "$YML" --prune
  echo ">>> Environment updated at $ENV_DIR"
fi

echo
echo ">>> To activate this environment, run:"
echo "    conda activate $ENV_DIR"
