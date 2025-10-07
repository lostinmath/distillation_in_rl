# Experiment Scripts

This directory contains bash scripts to reproduce all experiments from the adaptive RL research.

## Quick Start

```bash
# Test everything works (5 minutes)
./scripts/run_quick_test.sh

# Run complete study (4-6 hours)
./scripts/run_all_experiments.sh
```

## Individual Studies

### 1. Baseline Comparison (`run_baseline_comparison.sh`)
- **Purpose**: Compare all 7 scheduling strategies (main results)
- **Runtime**: ~2-3 hours
- **Strategies**: student_only, teacher_only, epsilon, epsilon_decreasing, alternating, teacher_then_student, reward_based
- **Environments**: CartPole-v1, Acrobot-v1
- **Seeds**: 3 different seeds for statistical significance

### 2. Hyperparameter Sensitivity (`run_hyperparameter_sensitivity.sh`)
- **Purpose**: Test sensitivity to key hyperparameters
- **Runtime**: ~1-2 hours
- **Parameters**: trust_length (3,5,7,10), learning_rate (1e-4 to 1e-3)
- **Focus**: Robustness of reward-based scheduling

### 3. Ablation Study (`run_ablation_study.sh`)
- **Purpose**: Isolate contribution of each component
- **Runtime**: ~1 hour
- **Ablations**: no_warmup, no_trust_threshold, simple_switching
- **Goal**: Understand which components matter most

### 4. Quick Test (`run_quick_test.sh`)
- **Purpose**: Fast integration test for development
- **Runtime**: ~5 minutes
- **Use cases**: CI/CD, debugging, quick verification

## Output Structure

Each script creates timestamped results directories:

```
results/
├── baseline_comparison_20241005_1420/
│   ├── cartpole/
│   │   ├── student_only/seed_42/
│   │   ├── teacher_only/seed_42/
│   │   └── reward_based/seed_42/
│   └── acrobot/...
├── hyperparameter_sensitivity_20241005_1630/
└── ablation_study_20241005_1800/
```

## Prerequisites

1. **Environment Setup**:
   ```bash
   uv sync  # Install dependencies
   ```

2. **GPU** (recommended):
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

3. **Disk Space**: ~10GB for complete study

## Analysis Pipeline

After running experiments:

```bash
# Generate comparison plots
python -m adaptive_rl.analysis.compare_strategies results/baseline_comparison_*/

# Create sensitivity analysis
python -m adaptive_rl.analysis.analyze_sensitivity results/hyperparameter_sensitivity_*/

# Ablation analysis
python -m adaptive_rl.analysis.ablation_analysis results/ablation_study_*/
```

## Expected Results

### Key Findings
- **Reward-based scheduling** shows 20-30% improvement in sample efficiency
- **Trust length = 5** optimal across environments
- **Warmup period** crucial for early stability

### Performance Baselines (CartPole-v1)
- student_only: ~200 episodes to solve
- teacher_only: Immediate success (upper bound)
- reward_based: ~100-150 episodes (target)

## Troubleshooting

### Common Issues

1. **GPU Memory**: Reduce `num_envs` in configs
2. **Import Errors**: Run `uv sync` first
3. **Permission Denied**: Run `chmod +x scripts/*.sh`
4. **Disk Space**: Monitor results/ directory size

### Debug Mode

For faster debugging, edit configs to use:
- `total_timesteps: 1000`
- `eval_freq: 500`
- `num_envs: 4`

## Publication

Results from these scripts directly support:
- Figure 1: Learning curves (`baseline_comparison`)
- Figure 2: Strategy comparison (`baseline_comparison`)
- Figure 3: Hyperparameter sensitivity (`hyperparameter_sensitivity`)
- Figure 4: Ablation results (`ablation_study`)
- Table 1: Final performance comparison (`baseline_comparison`)

## Contact

For questions about experimental setup or reproduction issues, refer to the main project documentation.