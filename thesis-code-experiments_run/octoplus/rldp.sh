env_id=PickCube-v1
demos=10 # 00 # number of demos to train on. 200_000
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_ms3.py configs/base_rlpd_ms3.yml \
  logger.exp_name="rlpd-${env_id}-state-${demos}_rl_demos-${seed}-walltime_efficient" logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=1_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/rl/trajectory.state.pd_joint_delta_pos.cpu.h5" \
  demo_type="rl" config_type="walltime_efficient"