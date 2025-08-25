#!/usr/bin/env bash

# setup_env.sh -- create/update conda env for this repo and verify it
# - Installs env under ./.envs/neuroharmonize-env (ignored by git)
# - Uses conda-forge only (avoids Anaconda ToS prompts)
# - Works on older conda (no --override-channels required)

set -euo pipefail

ENV_DIR="${ENV_DIR:-./.envs/neuroharmonize-env}"
YML="${YML:-environment.yml}"
VERIFY_SCRIPT="${VERIFY_SCRIPT:-verify_environment.bash}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Load your conda module or install Miniconda/Mambaforge." >&2
  exit 1
fi

if [[ ! -f "$YML" ]]; then
  echo "ERROR: $YML not found. Run this from the repo root." >&2
  exit 1
fi

mkdir -p "$(dirname "$ENV_DIR")"

# Temporary condarc → conda-forge only (no ToS)
TMPCONDARC="$(mktemp)"
trap 'rm -f "$TMPCONDARC"' EXIT
cat >"$TMPCONDARC" <<'RC'
channels:
  - conda-forge
channel_priority: strict
RC
export CONDARC="$TMPCONDARC"

echo ">>> Conda version: $(conda --version)"
echo ">>> Env path: $ENV_DIR"
echo ">>> Channels: conda-forge (strict)"

# Create or update
if conda env create -p "$ENV_DIR" -f "$YML"; then
  echo ">>> Environment created."
else
  echo ">>> Environment may exist; updating with --prune…"
  conda env update -p "$ENV_DIR" -f "$YML" --prune
  echo ">>> Environment updated."
fi

echo
echo ">>> Running verification…"
if [[ ! -f "$VERIFY_SCRIPT" ]]; then
  echo "WARNING: $VERIFY_SCRIPT not found; skipping verification."
  echo "         To verify manually: conda activate $ENV_DIR && bash verify_environment.bash"
  exit 0
fi

# Prefer 'conda run' if available (works even without activating in this shell)
if conda run --help >/dev/null 2>&1; then
  if conda run -p "$ENV_DIR" bash "$VERIFY_SCRIPT"; then
    echo ">>> Verification succeeded ✅"
  else
    echo ">>> Verification FAILED ❌"
    echo "Try: conda activate $ENV_DIR && bash $VERIFY_SCRIPT"
    exit 1
  fi
else
  # Fallback: tell user how to verify
  echo "NOTE: Your conda doesn’t support 'conda run'. Verify manually:"
  echo "      conda activate $ENV_DIR && bash $VERIFY_SCRIPT"
fi

echo
echo ">>> To activate this environment in a new shell:"
echo "    conda activate $ENV_DIR"
