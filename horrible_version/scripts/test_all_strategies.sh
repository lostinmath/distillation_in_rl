#!/bin/bash
# Quick test of all 7 scheduling strategies (5k timesteps each)

set -e

echo "🧪 Testing All 7 Scheduling Strategies"
echo "======================================"
echo "Quick validation runs (5k timesteps each)"
echo ""

# Define all strategies
declare -a STRATEGIES=(
    "student_only:Student-Only (PPO)"
    "teacher_only:Teacher-Only (Optimal)"
    "epsilon:Fixed Epsilon (50/50)"
    "epsilon_decreasing:Decreasing Epsilon"
    "alternating:Alternating (Interchangeably)"
    "teacher_then_student:Teacher-then-Student"
    "reward_based:🎯 Reward-Based (MAIN)"
)

# Create quick test config template
create_quick_config() {
    local strategy=$1
    local config_file="configs/experiments/quick/test_${strategy}.yaml"

    cat > "$config_file" << EOF
# Quick test for $strategy strategy

experiment:
  name: "quick_test_${strategy}"
  seed: 42
  device: "cpu"

environment:
  env_id: "CartPole-v1"
  num_envs: 4

algorithm:
  name: "ppo"
  learning_rate: 3e-4
  n_steps: 128
  batch_size: 32
  n_epochs: 5
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5

scheduler:
  name: "$strategy"
EOF

    # Add strategy-specific parameters
    case $strategy in
        "epsilon")
            echo "  epsilon: 0.5" >> "$config_file"
            ;;
        "epsilon_decreasing")
            echo "  epsilon_start: 0.9" >> "$config_file"
            echo "  epsilon_end: 0.1" >> "$config_file"
            echo "  epsilon_decay_steps: 3000" >> "$config_file"
            ;;
        "reward_based")
            echo "  trust_period: 3" >> "$config_file"
            echo "  performance_window: 2" >> "$config_file"
            ;;
        "teacher_then_student")
            echo "  switch_timestep: 2500" >> "$config_file"
            ;;
    esac

    # Add teacher config (except for student_only)
    if [ "$strategy" != "student_only" ]; then
        cat >> "$config_file" << EOF

teacher:
  name: "optimal"
EOF
    else
        echo "" >> "$config_file"
        echo "teacher: null" >> "$config_file"
    fi

    # Add training config
    cat >> "$config_file" << EOF

training:
  total_timesteps: 5000
  eval_freq: 2000
  checkpoint_freq: 5000
  save_freq: 5000

paths:
  log_dir: "logs/strategy_test/${strategy}"
  checkpoint_dir: "logs/strategy_test/${strategy}/checkpoints"
  results_dir: "logs/strategy_test/${strategy}/results"

tracker:
  backends: ["console", "csv"]
  console:
    enabled: true
    verbose: false
  csv:
    enabled: true
EOF

    echo "$config_file"
}

# Function to run strategy test
test_strategy() {
    local strategy=$1
    local strategy_name=$2
    local start_time=$(date +%s)

    echo ""
    echo "🔍 Testing: $strategy_name"
    echo "Strategy: $strategy"

    # Create config file
    local config_file=$(create_quick_config "$strategy")

    # Run test
    if uv run python train_modern.py "$config_file" --verbose > /dev/null 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "✅ PASSED: $strategy_name (${duration}s)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "❌ FAILED: $strategy_name (${duration}s)"
        return 1
    fi
}

# Create directories
mkdir -p configs/experiments/quick
mkdir -p logs/strategy_test

echo "Running strategy validation tests..."

# Test all strategies
passed=0
failed=0

for strategy_info in "${STRATEGIES[@]}"; do
    IFS=':' read -r strategy strategy_name <<< "$strategy_info"

    if test_strategy "$strategy" "$strategy_name"; then
        ((passed++))
    else
        ((failed++))
    fi
done

echo ""
echo "🎯 STRATEGY TEST RESULTS"
echo "======================="
echo "✅ Passed: $passed"
echo "❌ Failed: $failed"
echo "📊 Success rate: $((passed * 100 / (passed + failed)))%"

if [ $failed -eq 0 ]; then
    echo ""
    echo "🎉 ALL STRATEGIES WORKING!"
    echo "Ready for comprehensive experiments:"
    echo "   ./scripts/run_comprehensive_experiments.sh"
else
    echo ""
    echo "⚠️  Some strategies failed - check logs for details"
    exit 1
fi