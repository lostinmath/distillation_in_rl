#!/bin/bash
# Scientific baseline comparison: Student-only vs Reward-based scheduling

set -e

echo "🔬 Running Scientific Baseline Comparison"
echo "========================================"
echo "Comparing student-only (PPO) vs reward-based scheduling on CartPole and Acrobot"
echo ""

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="logs/scientific_comparison_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "📂 Results directory: $RESULTS_DIR"

# Save git info for reproducibility
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
echo "📝 Git commit: $GIT_COMMIT"
echo "$GIT_COMMIT" > "$RESULTS_DIR/git_commit.txt"

# Function to run experiment with error handling
run_experiment() {
    local config_file=$1
    local experiment_name=$2

    echo ""
    echo "🚀 Starting: $experiment_name"
    echo "Config: $config_file"

    if uv run python train_modern.py "$config_file" --verbose; then
        echo "✅ Completed: $experiment_name"
    else
        echo "❌ Failed: $experiment_name"
        return 1
    fi
}

# Run experiments in sequence
echo ""
echo "Running experiments sequentially for fair comparison..."

# Experiment 1: Pure PPO baseline
run_experiment "configs/experiments/scientific/01_student_only_cartpole.yaml" "Student-Only Baseline"

# Experiment 2: Reward-based scheduling (main contribution)
run_experiment "configs/experiments/scientific/02_reward_based_cartpole.yaml" "Reward-Based Scheduling"

# Experiment 3: Acrobot student-only (generalization test)
run_experiment "configs/experiments/comprehensive/acrobot_student_only.yaml" "Acrobot Student-Only"

# Experiment 4: Acrobot reward-based (generalization validation)
run_experiment "configs/experiments/comprehensive/acrobot_reward_based.yaml" "Acrobot Reward-Based"

echo ""
echo "🎉 Scientific comparison completed!"
echo ""
echo "📊 Results available in:"
echo "   - logs/scientific/01_student_only_cartpole/"
echo "   - logs/scientific/02_reward_based_cartpole/"
echo "   - logs/comprehensive/acrobot_student_only/"
echo "   - logs/comprehensive/acrobot_reward_based/"
echo ""
echo "🔍 Next steps:"
echo "   1. Compare learning curves in TensorBoard"
echo "   2. Analyze CSV logs for statistical significance"
echo "   3. Run sensitivity analysis with different seeds"