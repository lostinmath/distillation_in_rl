#!/bin/bash
# Run complete experimental study with all experiment groups
# This is the main script for reproducing all research results

set -e

echo "Running Complete Experimental Study"
echo "==================================="
echo "This will run all experiment groups sequentially:"
echo "  - Group 01: Baseline Study"
echo "  - Group 02: Scheduling Strategy Comparison"
echo ""
echo "Estimated total runtime: 2-4 hours"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Record start time
START_TIME=$(date +%s)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Study started at: $(date)"
echo ""

# Create main results directory
mkdir -p results/complete_study_${TIMESTAMP}
cd results/complete_study_${TIMESTAMP}

# Create study log
STUDY_LOG="study_log.txt"
echo "Complete Experimental Study - $(date)" > ${STUDY_LOG}
echo "========================================" >> ${STUDY_LOG}
echo "" >> ${STUDY_LOG}

# Run baseline study
echo "PHASE 1: Baseline Study"
echo "======================="
../../scripts/run_baseline_study.sh 2>&1 | tee -a ${STUDY_LOG}

echo "" | tee -a ${STUDY_LOG}
echo "PHASE 2: Scheduling Strategy Comparison" | tee -a ${STUDY_LOG}
echo "=======================================" | tee -a ${STUDY_LOG}
../../scripts/run_scheduling_study.sh 2>&1 | tee -a ${STUDY_LOG}

cd ../..

# Calculate runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(((RUNTIME % 3600) / 60))

echo ""
echo "Complete Experimental Study finished successfully"
echo "================================================="
echo "Total runtime: ${HOURS}h ${MINUTES}m"
echo "Results saved in: results/complete_study_${TIMESTAMP}/"
echo ""
echo "Next steps for analysis:"
echo "  1. Run statistical validation:"
echo "     pixi run python examples/run_statistical_validation.py"
echo "  2. Generate comparison plots:"
echo "     pixi run python notebooks/plots.ipynb"
echo "  3. Review results in: results/complete_study_${TIMESTAMP}/study_log.txt"