#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 15:06:44 2025

@author: andersonion (BJ Anderson)
"""

set -euo pipefail

ENV_DIR="./.envs/neuroharmonize-env"

echo ">>> Creating conda environment at $ENV_DIR"
mkdir -p "$(dirname "$ENV_DIR")"

if conda env create -f environment.yml --override-channels -p "$ENV_DIR"; then
    echo ">>> Environment created."
else
    echo ">>> Environment may already exist; updating instead."
    conda env update -f environment.yml --prune -p "$ENV_DIR"
fi

echo ""
echo ">>> To activate, run:"
echo "    conda activate $ENV_DIR"

