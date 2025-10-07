#!/bin/bash
# Robust experimental design with multiple seeds for statistical significance

set -e

echo "Robust Scientific Experiments"
echo "=============================="
echo "Multi-seed training for statistical robustness"
echo ""

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="logs/robust_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "Results: $RESULTS_DIR"

# Save git info for reproducibility
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
echo "Commit: $GIT_COMMIT"
echo "$GIT_COMMIT" > "$RESULTS_DIR/git_commit.txt"

# Define seeds for statistical robustness (mathematically interesting numbers)
SEEDS=(6 28 496 8128 153 371 407 1634 9474 54748)
# Perfect numbers (6, 28, 496, 8128): Equal the sum of their proper divisors
# Armstrong numbers (153, 371, 407, 1634, 9474, 54748): Equal the sum of their digits raised to the power of the number of digits
STRATEGIES=("student_only" "reward_based")

# Function to create config file for a specific seed and strategy
create_config() {
    local strategy=$1
    local seed=$2
    local config_file="configs/experiments/scientific_robust/${strategy}_seed${seed}.yaml"

    mkdir -p "configs/experiments/scientific_robust"

    cat > "$config_file" << EOF
# Scientific Experiment: ${strategy^} (Seed $seed)

experiment:
  name: "cartpole_${strategy}_seed${seed}"
  seed: $seed
  device: "cpu"

environment:
  env_id: "CartPole-v1"
  num_envs: 8

algorithm:
  name: "ppo"
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5

scheduler:
  name: "$strategy"
EOF

    # Add strategy-specific parameters
    if [ "$strategy" = "reward_based" ]; then
        cat >> "$config_file" << EOF
  trust_period: 5
  performance_window: 3

teacher:
  name: "optimal"
EOF
    else
        echo "" >> "$config_file"
        echo "teacher: null" >> "$config_file"
    fi

    # Add training configuration
    cat >> "$config_file" << EOF

training:
  total_timesteps: 500000  # Extended for convergence
  eval_freq: 5000          # More frequent evaluation
  checkpoint_freq: 50000
  save_freq: 100000

paths:
  log_dir: "logs/scientific_robust/cartpole_${strategy}_seed${seed}"
  checkpoint_dir: "logs/scientific_robust/cartpole_${strategy}_seed${seed}/checkpoints"
  results_dir: "logs/scientific_robust/cartpole_${strategy}_seed${seed}/results"

tracker:
  backends: ["tensorboard", "csv"]
  tensorboard:
    enabled: true
  csv:
    enabled: true
  console:
    enabled: false
EOF

    echo "$config_file"
}

# Function to run experiment with progress tracking
run_experiment() {
    local config_file=$1
    local experiment_name=$2
    local exp_num=$3
    local total_exps=$4
    local start_time=$(date +%s)

    printf "[%2d/%2d] %s " "$exp_num" "$total_exps" "$experiment_name"

    # Run experiment and capture output
    if uv run python train_modern.py "$config_file" 2>/dev/null | \
       grep -E "Progress:|episode/return" | \
       while IFS= read -r line; do
           if [[ $line == *"Progress:"* ]]; then
               progress=$(echo "$line" | grep -o '[0-9.]*%' | head -1)
               printf "\r[%2d/%2d] %s [%s]" "$exp_num" "$total_exps" "$experiment_name" "$progress"
           fi
       done; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        printf "\r[%2d/%2d] %s [DONE] %ds\n" "$exp_num" "$total_exps" "$experiment_name" "$duration"
        echo "$experiment_name,$config_file,success,$duration" >> "$RESULTS_DIR/experiment_log.csv"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        printf "\r[%2d/%2d] %s [FAIL] %ds\n" "$exp_num" "$total_exps" "$experiment_name" "$duration"
        echo "$experiment_name,$config_file,failed,$duration" >> "$RESULTS_DIR/experiment_log.csv"
        return 1
    fi
}

# Initialize experiment log
echo "experiment_name,config_file,status,duration_seconds" > "$RESULTS_DIR/experiment_log.csv"

total_experiments=$((${#SEEDS[@]} * ${#STRATEGIES[@]}))
current_experiment=0

echo ""
echo "Running $total_experiments experiments"
echo "Seeds: ${SEEDS[*]}"
echo "Strategies: ${STRATEGIES[*]}"
echo ""

# Run all experiments
for strategy in "${STRATEGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        current_experiment=$((current_experiment + 1))

        # Create config file
        config_file=$(create_config "$strategy" "$seed")

        # Run experiment with clean progress display
        experiment_name="${strategy}_s${seed}"
        run_experiment "$config_file" "$experiment_name" "$current_experiment" "$total_experiments"
    done
done

echo ""
echo "EXPERIMENTS COMPLETED"
echo "===================="
echo ""
echo "Results: $RESULTS_DIR"
echo "Log: $RESULTS_DIR/experiment_log.csv"
echo ""
echo "Next: uv run python analyze_robust_results.py"