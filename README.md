
# OctoPlus: Knowledge Distillation for Robotics

Originally developed for distillation with Octo transformer models, OctoPlus is a general-purpose distillation strategy that works with any teacher-student architecture in robotics RL.

The code is mostly based on the Master thesis.

The general Distillation method is in octoplus, 
and different environments are subpackages.
```
octoplus/
├── octoplus/           # Core distillation method (RL-agnostic)
├── octoplus-maniskill/ # ManiSkill-specific implementations
├── octoplus-dmcontrol/ # DeepMind Control Suite
```



## Installation

To create uv environment run

```bash
uv venv --python 3.12
```

To run pre-commit hooks run

```bash
pre-commit run --all-files
```
