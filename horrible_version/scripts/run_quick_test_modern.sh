#!/bin/bash
# Quick test to validate core functionality with modern architecture

set -e

echo "🧪 Running Quick Functionality Test (Modern Architecture)"
echo "========================================================"

# Check if we're in the right directory
if [ ! -f "train_modern.py" ]; then
    echo "❌ Error: train_modern.py not found. Run from project root."
    exit 1
fi

# Create logs directory
mkdir -p logs/quick_test

# Run quick test with validation only first
echo "1. Validating configuration..."
uv run python train_modern.py configs/experiments/quick/quick_test_cartpole.yaml --validate-only

echo ""
echo "2. Running dry run..."
uv run python train_modern.py configs/experiments/quick/quick_test_cartpole.yaml --dry-run

echo ""
echo "3. Running actual quick test (10k timesteps)..."
uv run python train_modern.py configs/experiments/quick/quick_test_cartpole.yaml --verbose

echo ""
echo "✅ Quick test completed!"
echo "📊 Check logs/quick_test/ for results"
echo ""
echo "🔬 Ready for scientific experiments! Run:"
echo "   ./scripts/run_baseline_comparison_modern.sh"