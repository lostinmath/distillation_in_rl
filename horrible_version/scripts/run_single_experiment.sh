#!/bin/bash
# Run a single experiment with proper setup and logging
# Usage: ./scripts/run_single_experiment.sh <group> <experiment_name>
# Example: ./scripts/run_single_experiment.sh 02_scheduling_comparison 02_001_reward_based_cartpole

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <group> <experiment_name>"
    echo ""
    echo "Examples:"
    echo "  $0 01_baseline_study 01_001_student_only_cartpole"
    echo "  $0 02_scheduling_comparison 02_001_reward_based_cartpole"
    echo ""
    echo "Available experiments:"
    echo "  Group 01 (Baseline Study):"
    echo "    - 01_001_student_only_cartpole"
    echo "    - 01_002_teacher_only_cartpole"
    echo ""
    echo "  Group 02 (Scheduling Comparison):"
    echo "    - 02_001_reward_based_cartpole"
    echo "    - 02_002_epsilon_05_cartpole"
    echo "    - 02_003_epsilon_decreasing_cartpole"
    exit 1
fi

GROUP=$1
EXPERIMENT=$2

# Check if we're in the right directory
if [ ! -f "src/adaptive_rl/train.py" ]; then
    echo "Error: Not in project root directory. Please run from project root."
    exit 1
fi

# Check if experiment config exists
CONFIG_PATH="configs/experiments/${GROUP}/${EXPERIMENT}.yaml"
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Experiment configuration not found: ${CONFIG_PATH}"
    exit 1
fi

echo "Running Single Experiment"
echo "========================="
echo "Group: ${GROUP}"
echo "Experiment: ${EXPERIMENT}"
echo "Config: ${CONFIG_PATH}"
echo ""

# Create results directory
RESULT_DIR="results/single_experiments/${GROUP}"
mkdir -p "${RESULT_DIR}"
cd "${RESULT_DIR}"

# Run the experiment
echo "Starting experiment at: $(date)"
echo ""

../../../run_experiment.sh \
    --config-path configs/experiments/${GROUP} \
    --config-name ${EXPERIMENT}

cd ../../..

echo ""
echo "Single experiment completed successfully"
echo "Results saved in: ${RESULT_DIR}/"
echo ""
echo "To analyze results:"
echo "  - Check tensorboard logs in the results directory"
echo "  - Review training metrics and final performance"