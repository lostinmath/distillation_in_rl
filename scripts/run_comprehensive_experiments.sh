#!/bin/bash
# Comprehensive experiment suite: All 7 strategies on CartPole and LunarLander

set -e

echo "🔬 Running Comprehensive Experiment Suite"
echo "========================================"
echo "Testing all 7 scheduling strategies on CartPole-v1"
echo "Key metrics: sample efficiency, final performance, variance"
echo ""

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="logs/comprehensive_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "📂 Results directory: $RESULTS_DIR"

# Save git info for reproducibility
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
echo "📝 Git commit: $GIT_COMMIT"
echo "$GIT_COMMIT" > "$RESULTS_DIR/git_commit.txt"

# Define all experiments
declare -a CARTPOLE_EXPERIMENTS=(
    "configs/experiments/comprehensive/cartpole_student_only.yaml:Student-Only (PPO Baseline)"
    "configs/experiments/comprehensive/cartpole_teacher_only.yaml:Teacher-Only (Optimal Policy)"
    "configs/experiments/comprehensive/cartpole_epsilon.yaml:Fixed Epsilon (50/50)"
    "configs/experiments/comprehensive/cartpole_epsilon_decreasing.yaml:Decreasing Epsilon"
    "configs/experiments/comprehensive/cartpole_interchangeably.yaml:Interchangeably"
    "configs/experiments/comprehensive/cartpole_teacher_then_student.yaml:Teacher-then-Student"
    "configs/experiments/comprehensive/cartpole_reward_based.yaml:🎯 Reward-Based (MAIN CONTRIBUTION)"
)

declare -a ACROBOT_EXPERIMENTS=(
    "configs/experiments/comprehensive/acrobot_student_only.yaml:Acrobot Student-Only"
    "configs/experiments/comprehensive/acrobot_reward_based.yaml:🎯 Acrobot Reward-Based"
)

# Function to run experiment with error handling
run_experiment() {
    local config_file=$1
    local experiment_name=$2
    local start_time=$(date +%s)

    echo ""
    echo "🚀 Starting: $experiment_name"
    echo "Config: $config_file"
    echo "Time: $(date)"

    if uv run python train_modern.py "$config_file" --verbose; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "✅ Completed: $experiment_name (${duration}s)"
        echo "$experiment_name,$config_file,success,$duration" >> "$RESULTS_DIR/experiment_log.csv"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "❌ Failed: $experiment_name (${duration}s)"
        echo "$experiment_name,$config_file,failed,$duration" >> "$RESULTS_DIR/experiment_log.csv"
        return 1
    fi
}

# Initialize experiment log
echo "experiment_name,config_file,status,duration_seconds" > "$RESULTS_DIR/experiment_log.csv"

echo ""
echo "🎯 CARTPOLE-V1 EXPERIMENTS"
echo "=========================="

# Run CartPole experiments
for experiment in "${CARTPOLE_EXPERIMENTS[@]}"; do
    IFS=':' read -r config_file experiment_name <<< "$experiment"
    run_experiment "$config_file" "$experiment_name"
done

echo ""
echo "🤸 ACROBOT-V1 EXPERIMENTS"
echo "========================="

# Run Acrobot experiments (key comparison)
for experiment in "${ACROBOT_EXPERIMENTS[@]}"; do
    IFS=':' read -r config_file experiment_name <<< "$experiment"
    run_experiment "$config_file" "$experiment_name"
done

echo ""
echo "🎉 COMPREHENSIVE EXPERIMENTS COMPLETED!"
echo "========================================="
echo ""
echo "📊 Results available in:"
echo "   - logs/comprehensive/ (individual experiment logs)"
echo "   - $RESULTS_DIR/experiment_log.csv (summary)"
echo ""
echo "🔍 Analysis next steps:"
echo "   1. Compare learning curves: tensorboard --logdir logs/comprehensive"
echo "   2. Statistical analysis of CSV files"
echo "   3. Generate comparison plots"
echo ""
echo "🎯 Key hypothesis to validate:"
echo "   Reward-based scheduling should show:"
echo "   ✓ Faster initial learning (sample efficiency)"
echo "   ✓ Lower variance across runs"
echo "   ✓ Better final performance than student-only"
echo "   ✓ Adaptive switching behavior in logs"