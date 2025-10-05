#!/bin/bash

# Quick Test Script - Fast verification that everything works
# Use for development, debugging, and CI/CD

set -e

echo "=== Running Quick Integration Test ==="
echo "Fast verification that all components work together"

RESULTS_DIR="results/quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"

# Save git commit ID for reproducibility
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
echo "Git commit: $GIT_COMMIT"
echo "Git branch: $GIT_BRANCH"

# Save git info to results directory
cat > "$RESULTS_DIR/git_info.txt" << EOF
Git Commit: $GIT_COMMIT
Git Branch: $GIT_BRANCH
Timestamp: $(date)
Command: $0 $@
EOF

# Test core strategies with minimal resources
STRATEGIES=("student_only" "teacher_only" "reward_based")
SEED=42

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "Testing strategy: $strategy"

    uv run adaptive-train \
        scheduler="$strategy" \
        experiment.seed="$SEED" \
        training.total_timesteps=1000 \
        training.eval_freq=500 \
        paths.log_dir="$RESULTS_DIR/${strategy}" \
        hydra.run.dir="$RESULTS_DIR/${strategy}"

    echo "✓ Strategy $strategy completed successfully"
done

echo ""
echo "=== Quick Test Complete ==="
echo "All core strategies working correctly!"
echo ""
echo "For full experiments, run:"
echo "- ./scripts/run_baseline_comparison.sh"
echo "- ./scripts/run_hyperparameter_sensitivity.sh"
echo "- ./scripts/run_ablation_study.sh"