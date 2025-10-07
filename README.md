# Adaptive Teacher-Student Scheduling for Reinforcement Learning

Python 3.10+ | Code style: black | Linting: ruff

## Overview

This project implements adaptive scheduling strategies for teacher-guided reinforcement learning. The core research question: **When should an RL agent follow teacher guidance versus explore independently?**

### Research Contribution

The main contribution is a reward-based scheduling strategy that adaptively determines when an RL agent should follow teacher guidance versus explore independently. The method monitors performance trends and switches policies when reward decreases are detected, leading to improved sample efficiency compared to fixed scheduling approaches.

## Installation

### Prerequisites

- Python 3.10+
- Git
- CUDA-capable GPU (recommended)

### Setup

```bash
# Install pixi package manager
curl -fsSL https://pixi.sh/install.sh | bash

# Clone repository
git clone https://github.com/username/distillation_in_rl.git
cd distillation_in_rl

# Install dependencies
pixi install

# Verify installation
pixi run python -c "import adaptive_rl; print('Installation successful')"
```

## Quick Start

### Running Experiments

```bash
# Run baseline comparison study
./scripts/run_baseline_comparison.sh

# Run quick integration test
./scripts/run_quick_test.sh

# Run specific experiment group
python train_modern.py --config-path configs/experiments/01_baseline_study --config-name 01_001_student_only_cartpole
```

### Training Individual Experiments

```bash
# Train with reward-based scheduling (main contribution)
pixi run python train_modern.py \
    environment=cartpole \
    scheduler=reward_based \
    teacher=optimal

# Train baseline (student only)
pixi run python train_modern.py \
    environment=cartpole \
    scheduler=student_only

# Train with different environments
pixi run python train_modern.py \
    environment=lunarlander \
    scheduler=reward_based \
    teacher=optimal
```

## Scheduling Strategies

This project implements seven scheduling strategies for combining teacher and student policies:

### 1. Reward-Based (Main Contribution)
Adaptive switching based on performance trend monitoring:
- Monitors reward over a trust period (default: 5 steps)
- Switches when current reward is less than previous reward
- Dynamic adaptation based on learning progress
- Demonstrates improved sample efficiency compared to fixed schedules

### 2. Epsilon Strategies
- **epsilon**: Fixed probability scheduling (e.g., 30% teacher, 70% student)
- **epsilon_decreasing**: Linear decay from 100% teacher to 0% teacher usage

### 3. Structured Schedules
- **interchangeably**: Alternates between teacher and student each iteration
- **teacher_then_student**: Bootstrap phase with teacher, then switch to student

### 4. Baseline Methods
- **student_only**: Pure reinforcement learning (PPO) without teacher guidance
- **teacher_only**: Pure imitation learning using only teacher actions

## Project Structure

```
distillation_in_rl/
├── src/adaptive_rl/               # Main package
│   ├── schedulers/                # Policy scheduling strategies
│   │   ├── reward_based.py        # Main contribution
│   │   ├── epsilon.py             # Epsilon-based strategies
│   │   └── simple.py              # Baseline strategies
│   ├── teachers/                  # Teacher policies
│   │   ├── optimal.py             # Hand-coded optimal policies
│   │   ├── random.py              # Random baseline
│   │   └── pretrained.py          # Saved model teachers
│   ├── envs/                      # Environment wrappers
│   ├── utils/                     # Utilities (logging, etc.)
│   └── validation/                # Statistical validation pipeline
├── configs/                       # Experiment configurations
│   ├── experiments/               # Numbered experiment series
│   │   ├── 01_baseline_study/     # Baseline performance bounds
│   │   ├── 02_scheduling_comparison/ # Main strategy comparison
│   │   ├── 03_ablation_study/     # Component analysis
│   │   └── 04_domain_generalization/ # Cross-environment validation
│   ├── algorithm/                 # PPO hyperparameters
│   ├── environment/               # CartPole, LunarLander configs
│   ├── scheduler/                 # All scheduling strategies
│   └── teacher/                   # Teacher configurations
├── scripts/                       # Experiment runner scripts
│   ├── run_baseline_comparison.sh # Strategy comparison
│   ├── run_quick_test.sh          # Fast verification
│   └── test_all_strategies.sh     # Comprehensive testing
├── docs/                          # Documentation
└── results/                       # Experiment outputs (auto-generated)
```

## Configuration System

Experiments use **Hydra** for configuration management:

```yaml
# configs/scheduler/reward_based.yaml
name: reward_based
trust_length: 5                    # Steps before evaluating switch
policy_trust_threshold: 0.6        # Confidence threshold
```

```yaml
# configs/experiments/02_scheduling_comparison/02_001_reward_based_cartpole.yaml
defaults:
  - base_experiment
  - environment: cartpole
  - algorithm: ppo
  - scheduler: reward_based
  - teacher: optimal

experiment_id: "02_001"
experiment_name: "02_001_reward_based_cartpole"
experiment_group: "02_scheduling_comparison"

seed: 42
total_timesteps: 50000
eval_frequency: 5000
```

## Expected Results

### Performance Benchmarks (CartPole-v1)
- **student_only**: ~200 episodes to solve (baseline)
- **teacher_only**: Immediate success (upper bound performance)
- **epsilon**: ~150 episodes (fixed probability baseline)
- **reward_based**: ~100-120 episodes (main contribution)

### Key Metrics
- Sample efficiency improvement over baseline methods
- Reduced learning variance across different random seeds
- Adaptive switching behavior based on performance trends

## Reproducibility & Git Tracking

All experiments automatically track git information for reproducibility:

```bash
# Each experiment creates git_info.txt
results/baseline_comparison_20241005_1420/git_info.txt
```

```
Git Commit: a1b2c3d4e5f6789...
Git Branch: main
Timestamp: 2024-10-05 14:20:33
Command: ./scripts/run_baseline_comparison.sh
```

## Analysis & Visualization

```bash
# Generate comparison plots
pixi run python -m adaptive_rl.analysis.compare_strategies results/baseline_comparison_*/

# Run statistical validation
pixi run python examples/run_statistical_validation.py

# Create publication figures
pixi run python notebooks/plots.ipynb
```

## Development

### Code Quality

```bash
# Format and lint
pixi run black src/
pixi run ruff check src/ --fix

# Type checking
pixi run mypy src/adaptive_rl

# Run tests
pixi run python tests/test_integration.py

# Quality checks
pixi run lint && pixi run format && pixi run type-check
```

### Adding New Components

#### New Scheduling Strategy
```python
# src/adaptive_rl/schedulers/my_strategy.py
from .base import PolicyScheduler

class MyScheduler(PolicyScheduler):
    def choose_policy_type(self, iteration, global_step, steps_since_reset, prev_reward):
        # Your scheduling logic here
        return ["student" if condition else "teacher" for _ in range(self.num_envs)]
```

#### New Teacher Policy
```python
# src/adaptive_rl/teachers/my_teacher.py
from .base import TeacherPolicy

class MyTeacher(TeacherPolicy):
    def act(self, obs):
        # Your teacher logic here
        return actions
```

## Scientific Validation

This implementation preserves the **exact scheduling logic** from the original thesis research:

```python
# Core reward-based switching logic (preserved exactly)
if prev_reward[i] < self.prev_prev_reward[i] and steps_taken_on_policy[i] >= trust_length:
    # Switch policy
    current_policy = "teacher" if current_policy == "student" else "student"
```

## Hardware Requirements

- **Minimum**: CPU, 4GB RAM, 5GB disk space
- **Recommended**: CUDA GPU, 16GB RAM, 20GB disk space
- **Runtime estimates**:
  - Quick test: 5 minutes
  - Single strategy: 30-60 minutes
  - Complete baseline study: 2-4 hours

## Citation

```bibtex
@mastersthesis{piscenco2024distillation,
  title={Adaptive Scheduling Strategies for Teacher-Guided Reinforcement Learning},
  author={Piscenco, Margarita},
  year={2024},
  school={University},
  note={Code available at: https://github.com/username/distillation_in_rl}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

- **Research Questions**: Issues on GitHub
- **Collaboration**: Pull requests welcome
- **Reproducibility Issues**: Check `git_info.txt` in results

---

**Research Contribution**: The reward-based scheduling strategy demonstrates improved sample efficiency by adaptively switching between teacher and student policies based on performance trend monitoring, providing a principled approach to teacher-student policy mixing in reinforcement learning.