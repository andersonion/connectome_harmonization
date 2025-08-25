#!/usr/bin/env bash

# setup_env.sh -- create/update conda env for this repo and verify it
# - Installs env under ./.envs/neuroharmonize-env (ignored by git)
# - Forces conda-forge only (no Anaconda ToS) by:
#     * setting default_channels: [] in a temp condarc
#     * using --override-channels when supported

set -euo pipefail

ENV_DIR="${ENV_DIR:-./.envs/neuroharmonize-env}"
YML="${YML:-environment.yml}"
VERIFY_SCRIPT="${VERIFY_SCRIPT:-verify_environment.bash}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Install Miniconda/Mambaforge or load your module." >&2
  exit 1
fi

if [[ ! -f "$YML" ]]; then
  echo "ERROR: $YML not found. Run from the repo root." >&2
  exit 1
fi

mkdir -p "$(dirname "$ENV_DIR")"

# Detect support for --override-channels (newer conda)
OC_FLAG=""
if conda env create --help 2>/dev/null | grep -q -- '--override-channels'; then
  OC_FLAG="--override-channels"
fi

# Temporary condarc: conda-forge ONLY, no defaults
TMPCONDARC="$(mktemp)"
trap 'rm -f "$TMPCONDARC"' EXIT
cat >"$TMPCONDARC" <<'RC'
channels:
  - conda-forge
channel_priority: strict
default_channels: []          # critical: removes repo.anaconda.com/defaults
RC
export CONDARC="$TMPCONDARC"

echo ">>> Conda: $(conda --version)"
echo ">>> Env path: $ENV_DIR"
echo ">>> Channels: conda-forge (strict), defaults disabled"
echo ">>> override-channels support: $([[ -n "$OC_FLAG" ]] && echo yes || echo no)"

# Create or update the env
if conda env create $OC_FLAG -p "$ENV_DIR" -f "$YML"; then
  echo ">>> Environment created."
else
  echo ">>> Environment may exist; updating with --prune…"
  conda env update $OC_FLAG -p "$ENV_DIR" -f "$YML" --prune
  echo ">>> Environment updated."
fi

echo
echo ">>> Running verification…"
if [[ ! -f "$VERIFY_SCRIPT" ]]; then
  echo "WARNING: $VERIFY_SCRIPT not found; skipping verification."
  echo "         To verify manually: conda activate $ENV_DIR && bash verify_environment.bash"
  exit 0
fi

# Prefer conda run (no need to activate)
if conda run --help >/dev/null 2>&1; then
  if conda run -p "$ENV_DIR" bash "$VERIFY_SCRIPT"; then
    echo ">>> Verification succeeded ✅"
  else
    echo ">>> Verification FAILED ❌"
    echo "Try: conda activate $ENV_DIR && bash $VERIFY_SCRIPT"
    exit 1
  fi
else
  echo "NOTE: 'conda run' not available. Verify manually:"
  echo "      conda activate $ENV_DIR && bash $VERIFY_SCRIPT"
fi

echo
echo ">>> To activate later:"
echo "    conda activate $ENV_DIR"
