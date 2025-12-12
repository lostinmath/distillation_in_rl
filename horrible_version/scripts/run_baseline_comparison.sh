#!/bin/bash

# Baseline Comparison Study - Compare all 7 scheduling strategies
# This reproduces the main experimental comparison from the paper

set -e

echo "=== Running Baseline Comparison Study ==="
echo "Comparing all 7 scheduling strategies on CartPole and Acrobot"

SEEDS=(42 123 456)
ENVIRONMENTS=("cartpole" "acrobot")
STRATEGIES=(
    "student_only"
    "teacher_only"
    "epsilon"
    "epsilon_decreasing"
    "alternating"
    "teacher_then_student"
    "reward_based"
)

RESULTS_DIR="results/baseline_comparison_$(date +%Y%m%d_%H%M%S)"
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

total_runs=$((${#SEEDS[@]} * ${#ENVIRONMENTS[@]} * ${#STRATEGIES[@]}))
current_run=0

for seed in "${SEEDS[@]}"; do
    for env in "${ENVIRONMENTS[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            current_run=$((current_run + 1))
            echo ""
            echo "[$current_run/$total_runs] Running: $env + $strategy (seed=$seed)"

            config_path="studies/baseline_comparison/${env}/${strategy}.yaml"

            # Override seed and output directory
            uv run python -m adaptive_rl.train \
                --config-path="$config_path" \
                experiment.seed="$seed" \
                paths.log_dir="$RESULTS_DIR/${env}/${strategy}/seed_${seed}" \
                hydra.run.dir="$RESULTS_DIR/${env}/${strategy}/seed_${seed}"

            echo "✓ Completed: $env + $strategy (seed=$seed)"
        done
    done
done

echo ""
echo "=== Baseline Comparison Study Complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Run analysis: python -m adaptive_rl.analysis.compare_strategies $RESULTS_DIR"
echo "2. Generate plots: python -m adaptive_rl.analysis.plot_results $RESULTS_DIR"
echo "3. Create summary table: python -m adaptive_rl.analysis.create_table $RESULTS_DIR"