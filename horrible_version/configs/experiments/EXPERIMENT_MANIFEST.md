# Experiment Manifest

## Overview
This document tracks all experimental configurations for the adaptive RL distillation research.

## Experiment ID System
**Format**: `{group}_{experiment}_{description}`
- **Group**: 2-digit experiment group (01, 02, 03, ...)
- **Experiment**: 3-digit experiment within group (001, 002, 003, ...)
- **Description**: Environment and method description

This ensures **no ID overlaps** across different experiment groups.

## Experiment Groups

### Group 01 - Baseline Study
**Purpose**: Establish performance bounds and baseline comparisons
**Status**: ✅ Configured

- `01_001_student_only_cartpole.yaml` - Pure student learning baseline (lower bound)
- `01_002_teacher_only_cartpole.yaml` - Pure teacher guidance (upper bound)

### Group 02 - Scheduling Strategy Comparison
**Purpose**: Compare different teacher-student scheduling approaches
**Status**: ✅ Configured

- `02_001_reward_based_cartpole.yaml` - **MAIN CONTRIBUTION**: Reward-trend adaptive scheduling
- `02_002_epsilon_05_cartpole.yaml` - Fixed 50% teacher usage baseline
- `02_003_epsilon_decreasing_cartpole.yaml` - Linear decay from 100% to 0% teacher usage
- `02_004_interchangeably_cartpole.yaml` - Alternate teacher/student each episode *(planned)*

### Group 03 - Ablation Study
**Purpose**: Validate hyperparameter choices and component contributions
**Status**: 📋 Planned

- `03_001_trust_length_3_cartpole.yaml` - Trust length = 3 steps
- `03_002_trust_length_5_cartpole.yaml` - Trust length = 5 steps (baseline)
- `03_003_trust_length_10_cartpole.yaml` - Trust length = 10 steps
- `03_004_patience_1_cartpole.yaml` - Patience = 1 consecutive decrease
- `03_005_patience_3_cartpole.yaml` - Patience = 3 consecutive decreases (baseline)
- `03_006_patience_5_cartpole.yaml` - Patience = 5 consecutive decreases

### Group 04 - Domain Generalization
**Purpose**: Validate approach across different environments
**Status**: 📋 Planned

- `04_001_reward_based_lunarlander.yaml` - Main contribution on LunarLander
- `04_002_epsilon_05_lunarlander.yaml` - Fixed epsilon baseline on LunarLander
- `04_003_cross_domain_teacher.yaml` - CartPole teacher on LunarLander environment

## Experimental Protocol

### Standard Configuration
- **Seeds**: 42, 123, 456, 789, 1011 (5 seeds for statistical significance)
- **Total Timesteps**: 50,000 (sufficient for CartPole convergence)
- **Evaluation Frequency**: Every 5,000 steps
- **Algorithm**: PPO with consistent hyperparameters

### Metrics Tracked
- Final performance (mean reward over evaluation)
- Sample efficiency (steps to reach 90% max performance)
- Teacher usage ratio
- Policy switch frequency
- Training stability (performance variance)

### Statistical Analysis
- Mann-Whitney U tests for pairwise comparisons
- Effect size calculations (Cohen's d)
- Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals

## Naming Convention
`{group}_{experiment}_{description}.yaml`

Where:
- `group`: 2-digit experiment group (01, 02, 03, ...)
- `experiment`: 3-digit sequential experiment within group (001, 002, 003, ...)
- `description`: Environment and method description (e.g., `reward_based_cartpole`)

**Examples**:
- `01_001_student_only_cartpole.yaml` - Group 01, Experiment 001
- `02_001_reward_based_cartpole.yaml` - Group 02, Experiment 001
- `03_001_trust_length_3_cartpole.yaml` - Group 03, Experiment 001

This system ensures **globally unique experiment IDs** with no overlap between groups.

## Usage
```bash
# Run single experiment
python train.py --config-path configs/experiments/01_baseline_study --config-name 01_001_student_only_cartpole

# Run experiment group with multiple seeds
python run_experiment_series.py --group 01_baseline_study --seeds 42,123,456,789,1011

# Run specific experiment with statistical validation
python run_statistical_experiment.py --experiment 02_001_reward_based_cartpole --seeds 42,123,456,789,1011
```