
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""
Simplified PPO training script for the Upkie pendulum environment (Task 2),
modeled after the student's working implementation.

Usage (from this directory):

    # 1) Launch the simulator from another terminal:
    #       ./start_simulation.sh
    #
    # 2) Run training:
    #       python upkie/train_pendulum_ppo.py
"""

import os
from pathlib import Path

import gymnasium as gym
import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_ID = "Upkie-Spine-Pendulum"
ENV_KWARGS = dict(
    frequency=200.0,
    regulate_frequency=False,
    disable_env_checker=True,
)

N_ENVS = 1  # single env like the student code
TOTAL_TIMESTEPS = 100_000
TENSORBOARD_LOG_DIR = Path("logs/pendulum_task2")
MODEL_SAVE_DIR = Path("models/pendulum_task2")
MODEL_SAVE_BASENAME = "ppo_upkie_pendulum"


def train() -> None:
    # Register Upkie environments
    upkie.envs.register()

    # Make sure directories exist
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Configure the environment
    # We use a single environment for simplicity (as in the student code).
    env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        env_kwargs=ENV_KWARGS,
    )

    # 2. Define the PPO Model
    # Hyperparameters closely follow the student's implementation.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(TENSORBOARD_LOG_DIR),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    # 3. Train
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 4. Save the model
    model_path = MODEL_SAVE_DIR / MODEL_SAVE_BASENAME
    model.save(str(model_path))
    print(f"Training finished. Model saved to {model_path}.zip")


if __name__ == "__main__":
    train()
