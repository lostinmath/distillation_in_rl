#!/bin/bash
# Clean wrapper script to run experiments with proper Python path
# Usage: ./run_experiment.sh [hydra args]

# Find the project root directory by looking for the script itself
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Check for available training scripts (prioritize real training)
TRAIN_REAL="${PROJECT_ROOT}/train_real.py"
TRAIN_SIMPLE="${PROJECT_ROOT}/train_simple.py"
TRAIN_MODERN="${PROJECT_ROOT}/scripts/train_modern.py"
TRAIN_MODULE="${PROJECT_ROOT}/src/adaptive_rl/train.py"

if [ -f "${TRAIN_REAL}" ]; then
    TRAIN_SCRIPT="${TRAIN_REAL}"
    echo "Using REAL training script: ${TRAIN_SCRIPT}"
elif [ -f "${TRAIN_SIMPLE}" ]; then
    TRAIN_SCRIPT="${TRAIN_SIMPLE}"
    echo "Using simple training script: ${TRAIN_SCRIPT}"
elif [ -f "${TRAIN_MODERN}" ]; then
    TRAIN_SCRIPT="${TRAIN_MODERN}"
    echo "Using modern training script: ${TRAIN_SCRIPT}"
elif [ -f "${TRAIN_MODULE}" ]; then
    TRAIN_SCRIPT="${TRAIN_MODULE}"
    echo "Using module training script: ${TRAIN_SCRIPT}"
else
    echo "Error: Cannot find training script in:"
    echo "  - ${TRAIN_REAL}"
    echo "  - ${TRAIN_SIMPLE}"
    echo "  - ${TRAIN_MODERN}"
    echo "  - ${TRAIN_MODULE}"
    exit 1
fi

# Change to project root and set Python path
cd "${PROJECT_ROOT}"
export PYTHONPATH="src:$PYTHONPATH"

# Ensure pixi is available - dynamically find it
if command -v pixi >/dev/null 2>&1; then
    echo "Using pixi from PATH"
elif [ -f "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
    echo "Using pixi from $HOME/.pixi/bin"
else
    echo "Error: pixi not found. Please install pixi."
    exit 1
fi

exec pixi run python "${TRAIN_SCRIPT}" "$@"