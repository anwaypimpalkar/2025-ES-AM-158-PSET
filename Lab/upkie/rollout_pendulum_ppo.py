#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import upkie.envs

# ======== EDIT THESE TWO PATHS IF NEEDED =========
RUN_DIR = Path("./models/pendulum_task2")  # Folder for this run
MODEL_PATH = RUN_DIR / "ppo_upkie_pendulum"
NORM_PATH = RUN_DIR / "vecnormalize.pkl"
# =================================================


def make_env():
    """Create a single Upkie pendulum environment, same settings as training."""
    upkie.envs.register()
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

    # Optionally load normalization stats if they exist
    if NORM_PATH.exists():
        env = VecNormalize.load(str(NORM_PATH), base_env)
        env.training = False        # eval mode
        env.norm_reward = False     # keep reward in original scale
        print(f"Loaded VecNormalize stats from {NORM_PATH}")
    else:
        env = base_env
        print("WARNING: vecnormalize.pkl not found, running without normalization")

    # Load trained policy
    print(f"Loading model from {MODEL_PATH}")
    model = PPO.load(str(MODEL_PATH), env=env, device="auto")

    print("Starting rollout for 5 episodes...")

    for episode in range(5):
        obs = env.reset()
        ep_return = 0.0
        ep_len = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            ep_return += float(rewards[0])
            ep_len += 1

            if bool(dones[0]):
                break

        print(f"[EVAL {episode+1}/5] Return = {ep_return:.3f}, Length = {ep_len} steps")

    # After all episodes, print final state of last rollout
    final_obs = obs[0]
    final_pitch = float(final_obs[0])
    final_pos = float(final_obs[1])
    print(f"Final State - Pitch: {final_pitch:.3f} rad, Position: {final_pos:.3f} m")

    env.close()


if __name__ == "__main__":
    main()