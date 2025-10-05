# Configuration Structure

This directory contains all experiment configurations for the distillation RL project.

## Directory Structure

```
configs/
├── base/                  # Base configurations
│   ├── default.yaml      # Default hyperparameters
│   ├── cartpole.yaml     # CartPole-specific settings
│   └── lunarlander.yaml  # LunarLander-specific settings
│
├── experiments/          # Experiment configurations
│   ├── baseline/        # Baseline experiments
│   │   ├── student_only.yaml
│   │   └── teacher_only.yaml
│   │
│   ├── scheduling/      # Scheduling strategy experiments
│   │   ├── epsilon.yaml
│   │   ├── epsilon_decreasing.yaml
│   │   ├── interchangeably.yaml
│   │   ├── teacher_then_student.yaml
│   │   └── reward_based.yaml
│   │
│   └── ablation/        # Ablation studies
│       ├── trust_length/
│       ├── epsilon_values/
│       └── network_size/
│
└── sweeps/              # Hyperparameter sweeps
    └── wandb/          # W&B sweep configurations
```

## Usage

### Running a single experiment:
```bash
distillation-train --config configs/experiments/scheduling/reward_based.yaml
```

### Overriding parameters:
```bash
distillation-train --config configs/base/default.yaml \
                   --env CartPole-v1 \
                   --strategy reward_based \
                   --iterations 1000
```

### Creating new experiments:
1. Start from a base config
2. Override specific parameters
3. Document the experiment purpose

## Config Inheritance

Configs can inherit from base configurations using the `defaults` key:
```yaml
defaults:
  - /base/default
  - /base/cartpole

# Override specific parameters
scheduler:
  strategy: reward_based
  trust_length: 10
```