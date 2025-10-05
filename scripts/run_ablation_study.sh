#!/bin/bash

# Ablation Study - Test individual components of reward-based scheduling
# Isolate the contribution of each component in the reward-based strategy

set -e

echo "=== Running Ablation Study ==="
echo "Testing individual components of reward-based scheduling"

SEEDS=(42 123 456)
RESULTS_DIR="results/ablation_study_$(date +%Y%m%d_%H%M%S)"
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

ABLATIONS=(
    "full_reward_based"     # Full implementation (baseline)
    "no_warmup"            # Remove warmup period
    "no_trust_threshold"   # Remove trust threshold
    "simple_switching"     # Simplified reward logic
)

total_runs=$((${#SEEDS[@]} * ${#ABLATIONS[@]}))
current_run=0

for seed in "${SEEDS[@]}"; do
    for ablation in "${ABLATIONS[@]}"; do
        current_run=$((current_run + 1))
        echo ""
        echo "[$current_run/$total_runs] Running ablation: $ablation (seed=$seed)"

        config_path="studies/ablation_study/${ablation}.yaml"

        uv run python -m adaptive_rl.train \
            --config-path="$config_path" \
            experiment.seed="$seed" \
            paths.log_dir="$RESULTS_DIR/${ablation}/seed_${seed}" \
            hydra.run.dir="$RESULTS_DIR/${ablation}/seed_${seed}"

        echo "✓ Completed ablation: $ablation (seed=$seed)"
    done
done

echo ""
echo "=== Ablation Study Complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Analysis:"
echo "1. Compare component contributions: python -m adaptive_rl.analysis.ablation_analysis $RESULTS_DIR"
echo "2. Visualize ablation results: python -m adaptive_rl.analysis.plot_ablations $RESULTS_DIR"
echo ""
echo "Key insights to look for:"
echo "- Impact of warmup period on early performance"
echo "- Effect of trust threshold on switching frequency"
echo "- Contribution of sophisticated reward logic vs simple switching"