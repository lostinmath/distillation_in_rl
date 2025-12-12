#!/bin/bash
# Quick launcher script for analysis tools

case "$1" in
    "hp-search")
        ./tools/experiments/run_hp_search.sh "${@:2}"
        ;;
    "comparison")
        ./tools/experiments/run_comparison_study.sh "${@:2}"
        ;;
    "analysis")
        pixi run python tools/analysis/simple_analysis.py "${@:2}"
        ;;
    *)
        echo "Usage: $0 {hp-search|comparison|analysis} [options]"
        echo ""
        echo "Commands:"
        echo "  hp-search    - Run hyperparameter search"
        echo "  comparison   - Run full comparison study"
        echo "  analysis     - Analyze existing results"
        echo ""
        echo "Examples:"
        echo "  $0 hp-search --focused"
        echo "  $0 comparison"
        echo "  $0 analysis"
        ;;
esac