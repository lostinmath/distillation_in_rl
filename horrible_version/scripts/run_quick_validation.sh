#!/bin/bash
# Quick validation test to ensure all components work correctly
# Runs short experiments to verify system functionality

set -e

echo "Running Quick Validation Test"
echo "============================="
echo "This test validates that all experiment components work correctly"
echo "with minimal runtime (1000 timesteps per experiment)"
echo ""

# Check if we're in the right directory
if [ ! -f "src/adaptive_rl/train.py" ]; then
    echo "Error: Not in project root directory. Please run from project root."
    exit 1
fi

# Create validation results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="results/validation_${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"
cd "${RESULT_DIR}"

echo "Results will be saved in: ${RESULT_DIR}/"
echo ""

# Test basic student-only training
echo "Test 1: Student-only baseline"
echo "-----------------------------"
../../run_experiment.sh \
    --config-path configs/experiments/01_baseline_study \
    --config-name 01_001_student_only_cartpole \
    total_timesteps=1000 \
    eval_frequency=500

echo ""
echo "Test 2: Teacher-only training"
echo "-----------------------------"
../../run_experiment.sh \
    --config-path configs/experiments/01_baseline_study \
    --config-name 01_002_teacher_only_cartpole \
    total_timesteps=1000 \
    eval_frequency=500

echo ""
echo "Test 3: Reward-based scheduling (main contribution)"
echo "--------------------------------------------------"
../../run_experiment.sh \
    --config-path configs/experiments/02_scheduling_comparison \
    --config-name 02_001_reward_based_cartpole \
    total_timesteps=1000 \
    eval_frequency=500

cd ../..

echo ""
echo "Quick validation completed successfully"
echo "======================================"
echo "All core components are working correctly"
echo "Results saved in: ${RESULT_DIR}/"
echo ""
echo "System is ready for full experiments. Run:"
echo "  - Single group: ./scripts/run_baseline_study.sh"
echo "  - Main comparison: ./scripts/run_scheduling_study.sh"
echo "  - Complete study: ./scripts/run_complete_study.sh"