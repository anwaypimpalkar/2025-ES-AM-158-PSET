# ES/AM 158 Upkie Lab — Environment Setup

This guide explains how to create a local Python environment, install Upkie, and run the simulator.

## Prerequisites
- Linux or macOS development (not Google Colab).
- Use a local editor such as VS Code.

## Installation (create and activate Conda environment)
```bash
conda create -n upkie python=3.10
conda activate upkie
pip install upkie
```

For more detail please refer to the pdf.

## Submission

For all tasks please write a short lab report (2-4 pages) and the rollout video of your policy in Task2 & Task3. If you want to also include your code, please update that on a public github repository and submit the link.

An example gif:

## Task2:
[Check the video here if not show up](Task2.mp4)

<video controls width="720" playsinline>
  <source src="Task2.mp4" type="video/mp4">
</video>

### Training a Task 2 PPO policy

1. Start the simulator (Spine + PyBullet) in a separate terminal:
   ```bash
   cd upkie
   ./start_simulation.sh
   ```
2. In another terminal (same Conda env), launch PPO training:
   ```bash
   cd upkie
   python train_pendulum_ppo.py \
     --total-timesteps 500000 \
     --save-dir ../models/task2 \
     --log-dir ../logs/task2 \
     --progress-bar
   ```
   Flags let you adjust horizons, vectorized env counts, normalization, checkpoints, etc. Models and VecNormalize statistics are stored under the run-specific folder inside `--save-dir`. Use `--resume-from <path>` to keep training from a previous checkpoint.

## Task3:
[Check the video here if not show up](Task3.mp4)

<video controls width="720" playsinline>
  <source src="Task3.mp4" type="video/mp4">
</video>
