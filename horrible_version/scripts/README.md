# Experiment Runner Scripts

This directory contains scripts to run all experiments for the adaptive teacher-student scheduling research.

## Quick Start

### 1. Validation Test (5 minutes)
```bash
./scripts/run_quick_validation.sh
```
Validates that all components work correctly with short test runs.

### 2. Individual Experiment Groups

#### Baseline Study (30-60 minutes)
```bash
./scripts/run_baseline_study.sh
```
Runs Group 01 experiments establishing performance bounds.

#### Scheduling Comparison (60-90 minutes)
```bash
./scripts/run_scheduling_study.sh
```
Runs Group 02 experiments comparing scheduling strategies (includes main contribution).

### 3. Complete Study (2-4 hours)
```bash
./scripts/run_complete_study.sh
```
Runs all experiment groups sequentially for complete reproduction.

## Advanced Usage

### Single Experiment
```bash
./scripts/run_single_experiment.sh <group> <experiment_name>

# Examples:
./scripts/run_single_experiment.sh 01_baseline_study 01_001_student_only_cartpole
./scripts/run_single_experiment.sh 02_scheduling_comparison 02_001_reward_based_cartpole
```

### Multi-Seed Statistical Runs
```bash
./scripts/run_multi_seed_experiment.sh <group> <experiment_name> [seeds]

# Examples:
./scripts/run_multi_seed_experiment.sh 02_scheduling_comparison 02_001_reward_based_cartpole
./scripts/run_multi_seed_experiment.sh 02_scheduling_comparison 02_001_reward_based_cartpole "42,123,456,789,1011"
```

## Available Experiments

### Group 01: Baseline Study
- `01_001_student_only_cartpole` - Pure student learning baseline
- `01_002_teacher_only_cartpole` - Pure teacher guidance (upper bound)

### Group 02: Scheduling Strategy Comparison
- `02_001_reward_based_cartpole` - **Main contribution**: Reward-based adaptive scheduling
- `02_002_epsilon_05_cartpole` - Fixed 50% teacher usage baseline
- `02_003_epsilon_decreasing_cartpole` - Linear decay from 100% to 0% teacher

## Results Structure

```
results/
├── validation_YYYYMMDD_HHMMSS/     # Quick validation tests
├── 01_baseline_study/              # Individual group results
├── 02_scheduling_comparison/       # Individual group results
├── complete_study_YYYYMMDD_HHMMSS/ # Complete study results
├── single_experiments/             # Individual experiment results
└── multi_seed/                     # Multi-seed statistical runs
```

## Prerequisites

1. **Environment Setup**:
   ```bash
   pixi install  # Install dependencies
   ```

2. **Project Structure**: Run from project root directory

3. **Hardware**:
   - Minimum: CPU, 4GB RAM
   - Recommended: GPU for faster training

## Troubleshooting

### Common Issues

1. **"Not in project root directory"**
   - Ensure you run scripts from the main project directory
   - Check that `src/adaptive_rl/train.py` exists

2. **Configuration not found**
   - Verify experiment config exists in `configs/experiments/`
   - Check experiment name spelling

3. **Import errors**
   - Run `pixi install` to ensure dependencies are installed
   - Check that `src/adaptive_rl` is in Python path

### Debug Mode
Add configuration overrides for faster debugging:
```bash
pixi run python src/adaptive_rl/train.py [config args] total_timesteps=1000 eval_frequency=500
```

## Analysis After Experiments

1. **Statistical Validation**:
   ```bash
   pixi run python examples/run_statistical_validation.py
   ```

2. **Generate Plots**:
   ```bash
   pixi run python notebooks/plots.ipynb
   ```

3. **Review Logs**:
   - TensorBoard logs in results directories
   - Training metrics in CSV files
   - Study logs for complete runs