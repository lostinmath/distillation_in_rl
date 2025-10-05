#!/bin/bash

# Hyperparameter Sensitivity Analysis
# Test sensitivity of reward-based scheduling to key hyperparameters

set -e

echo "=== Running Hyperparameter Sensitivity Analysis ==="
echo "Testing reward-based scheduling sensitivity to trust_length and learning_rate"

SEEDS=(42 123 456)
RESULTS_DIR="results/hyperparameter_sensitivity_$(date +%Y%m%d_%H%M%S)"
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

# Trust length sensitivity
echo ""
echo "=== Testing Trust Length Sensitivity ==="
TRUST_LENGTHS=(3 5 7 10)

for seed in "${SEEDS[@]}"; do
    for trust_length in "${TRUST_LENGTHS[@]}"; do
        echo "Running trust_length=$trust_length (seed=$seed)"

        uv run python -m adaptive_rl.train \
            --config-path="studies/hyperparameter_sensitivity/trust_length/trust_${trust_length}.yaml" \
            experiment.seed="$seed" \
            paths.log_dir="$RESULTS_DIR/trust_length/trust_${trust_length}/seed_${seed}" \
            hydra.run.dir="$RESULTS_DIR/trust_length/trust_${trust_length}/seed_${seed}"

        echo "✓ Completed trust_length=$trust_length (seed=$seed)"
    done
done

# Learning rate sensitivity
echo ""
echo "=== Testing Learning Rate Sensitivity ==="
LEARNING_RATES=("1e-4" "2.5e-4" "5e-4" "1e-3")

for seed in "${SEEDS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo "Running learning_rate=$lr (seed=$seed)"

        # Convert scientific notation for filename safety
        lr_safe=$(echo "$lr" | sed 's/e-4/e4/g' | sed 's/e-3/e3/g')

        uv run python -m adaptive_rl.train \
            --config-path="studies/hyperparameter_sensitivity/learning_rate/lr_${lr_safe}.yaml" \
            experiment.seed="$seed" \
            training.learning_rate="$lr" \
            paths.log_dir="$RESULTS_DIR/learning_rate/lr_${lr_safe}/seed_${seed}" \
            hydra.run.dir="$RESULTS_DIR/learning_rate/lr_${lr_safe}/seed_${seed}"

        echo "✓ Completed learning_rate=$lr (seed=$seed)"
    done
done

echo ""
echo "=== Hyperparameter Sensitivity Analysis Complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Analyze trust length sensitivity: python -m adaptive_rl.analysis.analyze_sensitivity $RESULTS_DIR/trust_length --param trust_length"
echo "2. Analyze learning rate sensitivity: python -m adaptive_rl.analysis.analyze_sensitivity $RESULTS_DIR/learning_rate --param learning_rate"
echo "3. Create sensitivity plots: python -m adaptive_rl.analysis.plot_sensitivity $RESULTS_DIR"