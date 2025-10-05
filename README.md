
# Distillation-RL: Teacher-Guided Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

This project implements adaptive scheduling strategies for teacher-guided reinforcement learning. The core research question: **When should an RL agent follow teacher guidance versus explore independently?**

### Key Innovation

**Reward-based scheduling**: An adaptive strategy that monitors performance trends and switches between teacher and student policies when performance degrades, improving sample efficiency in RL training.

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/lostinmath/distillation_in_rl.git
cd distillation_in_rl

# Create virtual environment and install
uv venv --python 3.10
uv pip install -e .

# Install with all optional dependencies
uv pip install -e ".[all]"
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Test Installation

```bash
# Test that all imports work
uv run distillation-test
```

### Train with Reward-Based Scheduling (Main Contribution)

```bash
# Train CartPole with reward-based scheduling
uv run distillation-train --config distillation_rl/configs/cartpole_reward_based.yaml

# Or specify parameters directly
uv run distillation-train \
    --env CartPole-v1 \
    --strategy reward_based \
    --teacher optimal \
    --iterations 300
```

### Compare All Scheduling Strategies

```bash
# Run baseline (pure PPO, no teacher)
uv run distillation-train --strategy student_only --env CartPole-v1

# Run with teacher only (pure imitation)
uv run distillation-train --strategy teacher_only --teacher optimal --env CartPole-v1

# Run with epsilon scheduling
uv run distillation-train --strategy epsilon --epsilon 0.3 --teacher optimal --env CartPole-v1

# Run with decreasing epsilon
uv run distillation-train --strategy epsilon_decreasing --teacher optimal --env CartPole-v1

# Run with reward-based switching (main contribution)
uv run distillation-train --strategy reward_based --teacher optimal --env CartPole-v1
```

## Scheduling Strategies

### 1. **Reward-Based (Main Contribution)**
Monitors reward trends and switches policies adaptively:
- Tracks performance over a trust period
- Switches when `current_reward < previous_reward`
- Improves sample efficiency significantly

### 2. **Epsilon-Based**
- `epsilon`: Fixed probability of using teacher
- `epsilon_decreasing`: Linearly decreasing probability

### 3. **Simple Baselines**
- `student_only`: Pure RL without teacher
- `teacher_only`: Pure imitation learning
- `alternating`: Switch every iteration
- `teacher_then_student`: Bootstrap then switch

## Configuration

Experiments are configured via YAML files in `distillation_rl/configs/`:

```yaml
# Example: cartpole_reward_based.yaml
experiment:
  name: "cartpole_reward_based"
  seed: 42

environment:
  env_id: "CartPole-v1"
  num_envs: 8

scheduler:
  strategy: "reward_based"
  trust_length: 5  # Steps before evaluating switch

teacher:
  type: "optimal"  # Hand-coded optimal policy
```

## Project Structure

```
distillation_rl/
├── schedulers/       # Scheduling strategies
│   ├── reward_based.py  # Main contribution
│   ├── epsilon.py       # Epsilon-based strategies
│   └── simple.py        # Baseline strategies
├── teachers/         # Teacher policies
│   ├── optimal.py       # Hand-coded optimal
│   ├── random.py        # Random baseline
│   └── pretrained.py    # Load saved models
├── core/             # Core algorithms
│   └── ppo.py          # PPO implementation
├── configs/          # Experiment configurations
└── utils/            # Utilities
```

## Development

### Code Quality

```bash
# Format code
uv run black distillation_rl/
uv run ruff check distillation_rl/ --fix

# Type checking
uv run mypy distillation_rl/

# Run tests
uv run pytest

# Run all checks
uv run pre-commit run --all-files
```

### Adding New Scheduling Strategies

1. Create new scheduler in `distillation_rl/schedulers/`
2. Inherit from `PolicyScheduler` base class
3. Implement `choose_policy_type()` method
4. Register in `SCHEDULERS` dictionary

### Adding New Teachers

1. Create new teacher in `distillation_rl/teachers/`
2. Inherit from `TeacherPolicy` base class
3. Implement `act()` method
4. Register in `TEACHER_TYPES` dictionary

## Results

Expected performance on CartPole-v1:
- **student_only**: ~200 episodes to solve
- **teacher_only**: Immediate success (optimal teacher)
- **reward_based**: 50-100 episodes (2-4x improvement)

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{piscenco2024distillation,
  title={Adaptive Scheduling Strategies for Teacher-Guided Reinforcement Learning},
  author={Piscenco, Margarita},
  year={2024},
  school={Your University}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on research conducted for master's thesis
- Uses Gymnasium environments for benchmarking
- PPO implementation adapted from CleanRL
