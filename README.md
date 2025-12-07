# PPO HalfCheetah Training

This repository contains a Proximal Policy Optimization (PPO) implementation for training on the HalfCheetah-v4 environment from Gymnasium.

## Setup Instructions

### 1. Create a new conda environment with Python 3.11

```bash
conda create -n ppo_env python=3.11
```

### 2. Activate the environment

```bash
conda activate ppo_env
```

### 3. Install required dependencies

```bash
pip install "gymnasium[mujoco]" numpy torch matplotlib
```

### 4. Run the training script

```bash
python ppo.py
```

## Output

The training script will:
- Train a PPO agent on HalfCheetah-v4 for 1000 episodes
- Save model checkpoints every 50 episodes to the `saved_models/` directory
- Generate and save training plots to the `plots/` directory:
  - `training_metrics.png` - Combined plot with rewards, KL divergence, and entropy
  - `rewards.png` - Episode rewards over training
  - `kl_divergence.png` - KL divergence over training
  - `entropy.png` - Policy entropy over training

### 5. Run the testing script

```bash
python test.py
```
