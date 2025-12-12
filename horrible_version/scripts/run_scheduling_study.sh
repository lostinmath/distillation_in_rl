#!/bin/bash
# Run Group 02: Scheduling Strategy Comparison experiments
# Compares different teacher-student scheduling approaches including main contribution

set -e

echo "Running Scheduling Strategy Comparison (Group 02)"
echo "================================================="

# Find project root directory by looking for key files
find_project_root() {
    local current_dir="$(pwd)"
    while [ "$current_dir" != "/" ]; do
        if [ -f "$current_dir/run_experiment.sh" ] && [ -d "$current_dir/src/adaptive_rl" ]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
    done
    return 1
}

PROJECT_ROOT=$(find_project_root)
if [ -z "$PROJECT_ROOT" ]; then
    echo "Error: Cannot find project root directory with run_experiment.sh and src/adaptive_rl/"
    exit 1
fi

echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Create results directory
mkdir -p results/02_scheduling_comparison
cd results/02_scheduling_comparison

echo ""
echo "Experiment 02.001: Reward-based scheduling (MAIN CONTRIBUTION)"
echo "--------------------------------------------------------------"
"$PROJECT_ROOT/run_experiment.sh" \
    --config-path configs/experiments/02_scheduling_comparison \
    --config-name 02_001_reward_based_cartpole

echo ""
echo "Experiment 02.002: Fixed epsilon (50%) baseline"
echo "-----------------------------------------------"
"$PROJECT_ROOT/run_experiment.sh" \
    --config-path configs/experiments/02_scheduling_comparison \
    --config-name 02_002_epsilon_05_cartpole

echo ""
echo "Experiment 02.003: Epsilon decreasing baseline"
echo "----------------------------------------------"
"$PROJECT_ROOT/run_experiment.sh" \
    --config-path configs/experiments/02_scheduling_comparison \
    --config-name 02_003_epsilon_decreasing_cartpole

cd ../..

echo ""
echo "Scheduling Strategy Comparison completed successfully"
echo "Results saved in: results/02_scheduling_comparison/"
echo ""
echo "Analysis suggestions:"
echo "  - Compare sample efficiency across methods"
echo "  - Analyze teacher usage patterns"
echo "  - Run statistical validation: pixi run python examples/run_statistical_validation.py"