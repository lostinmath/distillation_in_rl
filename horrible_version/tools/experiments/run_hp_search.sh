#!/bin/bash
# Hyperparameter search launcher script

set -e

echo "🔬 Hyperparameter Search for Reward-Based Scheduling"
echo "===================================================="

# Set up environment - dynamically find pixi
if command -v pixi >/dev/null 2>&1; then
    echo "✅ pixi found in PATH"
elif [ -f "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
    echo "✅ pixi added to PATH from $HOME/.pixi/bin"
else
    echo "❌ pixi not found. Please install pixi or add it to PATH"
    exit 1
fi

# Default parameters
SEARCH_TYPE="focused"
MAX_WORKERS=4
MAX_EXPERIMENTS=""
TIME_LIMIT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --full)
      SEARCH_TYPE="full"
      echo "🎯 Using FULL search space (thousands of combinations)"
      shift
      ;;
    --focused)
      SEARCH_TYPE="focused"
      echo "🎯 Using FOCUSED search space (~12 strategic combinations)"
      shift
      ;;
    --workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --max-experiments)
      MAX_EXPERIMENTS="$2"
      shift 2
      ;;
    --time-limit)
      TIME_LIMIT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--full|--focused] [--workers N] [--max-experiments N] [--time-limit HOURS]"
      exit 1
      ;;
  esac
done

echo "⚡ Parallel workers: $MAX_WORKERS"
echo "📊 Search type: $SEARCH_TYPE"

if [[ -n "$MAX_EXPERIMENTS" ]]; then
    echo "🎲 Max experiments: $MAX_EXPERIMENTS"
fi

if [[ -n "$TIME_LIMIT" ]]; then
    echo "⏰ Time limit: ${TIME_LIMIT}h"
fi

echo ""
echo "Starting hyperparameter search..."
echo "This will run multiple experiments in parallel."
echo "Results will be saved with timestamp."
echo ""

# Build command
CMD="pixi run python tools/experiments/hyperparameter_search.py --search-type $SEARCH_TYPE --max-workers $MAX_WORKERS"

if [[ -n "$MAX_EXPERIMENTS" ]]; then
    CMD="$CMD --max-experiments $MAX_EXPERIMENTS"
fi

if [[ -n "$TIME_LIMIT" ]]; then
    CMD="$CMD --time-limit $TIME_LIMIT"
fi

echo "Command: $CMD"
echo ""

# Run the search
eval $CMD

echo ""
echo "🎉 Hyperparameter search completed!"
echo "Check the results/ directory for detailed analysis."