#!/bin/bash

# Master Experiment Runner - Run complete experimental study
# This reproduces all experiments from the paper in the correct order

set -e

echo "=============================================="
echo "  RL Distillation - Complete Experimental Study"
echo "  Reproducing: Teacher-Guided Reinforcement Learning"
echo "  with Adaptive Scheduling Strategies"
echo "=============================================="
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
STUDY_DIR="results/complete_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$STUDY_DIR"

echo "Study results will be saved to: $STUDY_DIR"

# Save git commit ID for reproducibility
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
echo "Git commit: $GIT_COMMIT"
echo "Git branch: $GIT_BRANCH"

# Save git info to study directory
cat > "$STUDY_DIR/git_info.txt" << EOF
Git Commit: $GIT_COMMIT
Git Branch: $GIT_BRANCH
Timestamp: $(date)
Command: $0 $@
Study Type: Complete Experimental Study
EOF

echo "Start time: $(date)"
echo ""

# Check prerequisites
echo "=== Checking Prerequisites ==="
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Please install uv first."
    exit 1
fi

if ! uv run python -c "import adaptive_rl" 2>/dev/null; then
    echo "❌ adaptive_rl package not importable. Run 'uv sync' first."
    exit 1
fi

echo "✅ Prerequisites satisfied"
echo ""

# Estimate total time
echo "=== Time Estimation ==="
echo "Estimated total runtime: ~4-6 hours (depending on hardware)"
echo "- Baseline comparison: ~2-3 hours (7 strategies × 2 envs × 3 seeds)"
echo "- Hyperparameter sensitivity: ~1-2 hours"  
echo "- Ablation study: ~1 hour"
echo ""

read -p "Continue with full experimental study? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Study cancelled. Run individual scripts for faster testing:"
    echo "- ./scripts/run_quick_test.sh (5 minutes)"
    echo "- ./scripts/run_baseline_comparison.sh (2-3 hours)"
    exit 0
fi

echo ""
echo "=== STUDY 1: Baseline Comparison ==="
echo "Comparing all 7 scheduling strategies (MAIN RESULTS)"
./scripts/run_baseline_comparison.sh
echo "✅ Baseline comparison complete"

echo ""
echo "=== STUDY 2: Hyperparameter Sensitivity ==="
echo "Testing sensitivity to key hyperparameters"
./scripts/run_hyperparameter_sensitivity.sh
echo "✅ Hyperparameter sensitivity complete"

echo ""
echo "=== STUDY 3: Ablation Study ==="
echo "Testing individual components of reward-based scheduling"
./scripts/run_ablation_study.sh
echo "✅ Ablation study complete"

echo ""
echo "=============================================="
echo "  COMPLETE EXPERIMENTAL STUDY FINISHED"
echo "=============================================="
echo ""
echo "Results saved to individual timestamped directories"
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "1. Run analysis: python -m adaptive_rl.analysis.compare_all_studies"
echo "2. Generate paper plots: python -m adaptive_rl.analysis.create_figures"
echo "3. Create summary table: python -m adaptive_rl.analysis.create_table"
