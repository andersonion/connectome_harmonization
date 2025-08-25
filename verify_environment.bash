#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 14:38:34 2025

@author: andersonion (BJ Anderson)
"""

python - <<'PY'
import sys, numpy, pandas, sklearn, matplotlib, patsy
print("OK ✅")
print("Python:", sys.version.split()[0])
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
import importlib
for m in ("neuroHarmonize","neuroCombat"):
    print(m, "import:", "OK" if importlib.util.find_spec(m) else "MISSING")
PY
