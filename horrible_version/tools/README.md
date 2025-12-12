# Tools Directory

This directory contains analysis and experimental tools for the project.

## Structure

### `analysis/`
- `analyze_results.py` - Comprehensive analysis pipeline for experiment results
- `simple_analysis.py` - Quick analysis script for comparison studies

### `experiments/`
- `hyperparameter_search.py` - Full hyperparameter optimization
- `simple_hp_search.py` - Focused hyperparameter search
- `profiled_hp_search.py` - Resource-profiled hyperparameter search
- `run_comparison_study.sh` - Automated comparison study runner
- `run_hp_search.sh` - Hyperparameter search launcher

## Usage

### Run hyperparameter search:
```bash
./tools/experiments/run_hp_search.sh --focused
```

### Analyze results:
```bash
pixi run python tools/analysis/simple_analysis.py
```

### Full comparison study:
```bash
./tools/experiments/run_comparison_study.sh
```