#!/bin/bash
# Run Group 01: Baseline Study experiments
# Establishes performance bounds with student-only and teacher-only baselines

set -e

echo "Running Baseline Study (Group 01)"
echo "=================================="

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
mkdir -p results/01_baseline_study
cd results/01_baseline_study

echo ""
echo "Experiment 01.001: Student-only baseline (CartPole)"
echo "---------------------------------------------------"
"$PROJECT_ROOT/run_experiment.sh" \
    --config-path configs/experiments/01_baseline_study \
    --config-name 01_001_student_only_cartpole

echo ""
echo "Experiment 01.002: Teacher-only upper bound (CartPole)"
echo "------------------------------------------------------"
"$PROJECT_ROOT/run_experiment.sh" \
    --config-path configs/experiments/01_baseline_study \
    --config-name 01_002_teacher_only_cartpole

cd ../..

echo ""
echo "Baseline Study completed successfully"
echo "Results saved in: results/01_baseline_study/"
echo ""
echo "Next steps:"
echo "  - Review baseline performance bounds"
echo "  - Run scheduling comparison: ./scripts/run_scheduling_study.sh"