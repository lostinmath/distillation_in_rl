#!/bin/bash
# Run experiment with multiple seeds for statistical significance
# Usage: ./scripts/run_multi_seed_experiment.sh <group> <experiment_name> [seeds]
# Example: ./scripts/run_multi_seed_experiment.sh 02_scheduling_comparison 02_001_reward_based_cartpole "42,123,456"

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <group> <experiment_name> [seeds]"
    echo ""
    echo "Examples:"
    echo "  $0 02_scheduling_comparison 02_001_reward_based_cartpole"
    echo "  $0 02_scheduling_comparison 02_001_reward_based_cartpole \"42,123,456,789,1011\""
    echo ""
    echo "Default seeds: 42,123,456,789,1011 (5 seeds for statistical significance)"
    exit 1
fi

GROUP=$1
EXPERIMENT=$2
SEEDS=${3:-"42,123,456,789,1011"}  # Default to 5 seeds

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

# Convert seeds string to array
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

echo "Running Multi-Seed Experiment"
echo "============================="
echo "Group: ${GROUP}"
echo "Experiment: ${EXPERIMENT}"
echo "Seeds: ${SEEDS}"
echo "Number of runs: ${#SEED_ARRAY[@]}"
echo ""

# Create results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="results/multi_seed/${GROUP}/${EXPERIMENT}_${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

echo "Results will be saved in: ${RESULT_DIR}/"
echo ""

# Run experiment for each seed
for i in "${!SEED_ARRAY[@]}"; do
    SEED=${SEED_ARRAY[$i]}
    RUN_NUM=$((i + 1))

    echo "Run ${RUN_NUM}/${#SEED_ARRAY[@]}: Seed ${SEED}"
    echo "$(printf '=%.0s' {1..40})"

    # Create seed-specific directory
    SEED_DIR="${RESULT_DIR}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"
    cd "${SEED_DIR}"

    # Run with specific seed
    ../../../../run_experiment.sh \
        --config-path configs/experiments/${GROUP} \
        --config-name ${EXPERIMENT} \
        seed=${SEED}

    cd ../../../..

    echo "Run ${RUN_NUM} completed"
    echo ""
done

echo "Multi-seed experiment completed successfully"
echo "All ${#SEED_ARRAY[@]} runs finished"
echo "Results saved in: ${RESULT_DIR}/"
echo ""
echo "Next steps for statistical analysis:"
echo "  1. Run statistical validation on results:"
echo "     pixi run python src/adaptive_rl/validation/experiment_runner.py \\"
echo "       --results-dir ${RESULT_DIR}"
echo "  2. Generate aggregate analysis:"
echo "     pixi run python notebooks/analyze_multi_seed_results.ipynb"