#!/bin/bash
# Automated comparison study runner
set -e

echo "🧪 Starting Comprehensive Scheduling Strategy Comparison"
echo "======================================================="

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDY_DIR="results/comparison_study_$TIMESTAMP"
mkdir -p "$STUDY_DIR"

echo "📂 Results will be saved to: $STUDY_DIR"

# Define experiments to run
declare -a EXPERIMENTS=(
    "cartpole_student_only:Student-Only Baseline"
    "cartpole_teacher_only:Teacher-Only Upper Bound"
    "cartpole_reward_based:Reward-Based (Main Contribution)"
    "cartpole_epsilon:Fixed Epsilon (50%)"
    "cartpole_epsilon_decreasing:Decreasing Epsilon"
    "cartpole_interchangeably:Interchangeably"
)

# Run each experiment
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config_name display_name <<< "$experiment"

    echo ""
    echo "🚀 Running: $display_name"
    echo "   Config: $config_name"
    echo "   Time: $(date)"

    start_time=$(date +%s)

    if timeout 300s ./run_experiment.sh \
        --config-path configs/experiments/comprehensive \
        --config-name "$config_name" > "$STUDY_DIR/${config_name}_output.log" 2>&1; then

        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "   ✅ Completed in ${duration}s"

        # Extract key metrics from log
        final_perf=$(tail -10 "$STUDY_DIR/${config_name}_output.log" | grep "Final performance:" | awk '{print $3}' || echo "N/A")
        echo "   📊 Final Performance: $final_perf"

    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "   ❌ Failed after ${duration}s"
        final_perf="Failed"
    fi

    # Log to summary CSV
    echo "$config_name,$display_name,$final_perf,$duration,$(date)" >> "$STUDY_DIR/summary.csv"
done

echo ""
echo "🔬 Running Analysis Pipeline..."
echo "================================"

# Run analysis
if python analyze_results.py --results-dir logs --output-dir "$STUDY_DIR/analysis"; then
    echo "✅ Analysis complete!"
    echo ""
    echo "📄 View detailed report: $STUDY_DIR/analysis/comparison_report.md"
    echo "📊 View plots: $STUDY_DIR/analysis/strategy_comparison.png"
    echo "📋 View summary: $STUDY_DIR/summary.csv"
else
    echo "⚠️  Analysis failed, but raw results are available in $STUDY_DIR"
fi

echo ""
echo "🎉 Comparison study complete!"
echo "Results directory: $STUDY_DIR"