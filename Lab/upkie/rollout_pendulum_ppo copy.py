#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import upkie.envs

# ======== EDIT THESE TWO PATHS =========
RUN_DIR = Path("../models/task2/ppo_pendulum_lqr_20251123-144452")  # <-- your folder
MODEL_PATH = RUN_DIR / "best" / "best_model.zip"
NORM_PATH = RUN_DIR / "vecnormalize.pkl"
# =======================================


def make_env():
    upkie.envs.register()
    # Same env ID and options as training
    env = gym.make(
        "Upkie-Spine-Pendulum",
        frequency=200.0,
        regulate_frequency=False,
        disable_env_checker=True,
    )
    return env


def main():
    # Base env + VecEnv wrapper
    base_env = DummyVecEnv([make_env])

    # Load normalization stats
    if NORM_PATH.exists():
        env = VecNormalize.load(str(NORM_PATH), base_env)
        env.training = False       # eval mode
        env.norm_reward = False    # keep reward in original scale
        print(f"Loaded VecNormalize stats from {NORM_PATH}")
    else:
        env = base_env
        print("WARNING: vecnormalize.pkl not found, running without normalization")

    # Load trained policy
    print(f"Loading model from {MODEL_PATH}")
    model = PPO.load(str(MODEL_PATH), env=env, device="auto")

    obs = env.reset()
    step = 0
    try:
        while True:
            # Deterministic policy for visualization
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            step += 1

            # If episode ended, reset
            if dones[0]:
                print(f"Episode ended at step {step}, resetting.")
                obs = env.reset()
                step = 0

            # Sleep a bit so sim/viewer isn’t insanely fast
            time.sleep(0.0)
    except KeyboardInterrupt:
        print("Stopping rollout.")
    finally:
        env.close()


if __name__ == "__main__":
    main()