#!/usr/bin/env bash

set -euo pipefail

python - <<'PY'
import sys, importlib, numpy, pandas, sklearn, matplotlib, patsy

print("Python:", sys.version.split()[0])
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)

missing = []
for m in ("neuroHarmonize","neuroCombat"):
    ok = importlib.util.find_spec(m) is not None
    print(f"{m} import:", "OK" if ok else "MISSING")
    if not ok:
        missing.append(m)

if missing:
    print("\nERROR: Missing packages:", ", ".join(missing))
    sys.exit(1)
else:
    print("\nAll required packages are present. ✅")
PY
