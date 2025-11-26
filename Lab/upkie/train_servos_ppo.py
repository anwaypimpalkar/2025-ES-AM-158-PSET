#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PPO training script for the full Upkie-Servos model (Task 3).
#
# Usage (from the Lab directory, with the simulator running via ./start_simulation.sh):
#
#   conda activate upkie
#   python upkie/train_servos_ppo.py
#
# After training finishes, you can visualize with:
#
#   python upkie/rollout_policy_servos.py \
#       --model ./models/servos_best/best_model.zip \
#       --episodes 5 --deterministic
#

from pathlib import Path

import gymnasium as gym
import numpy as np
import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Reuse wrappers from rollout
from rollout_policy_servos import ServoVelActionWrapper, ServoObsFlattenWrapper, FallTerminationWrapper
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_ENVS = 1                    # single environment
TOTAL_TIMESTEPS = 300_000     # you can increase later (e.g., 1_000_000)
FREQUENCY_HZ = 200.0
MAX_STEPS_PER_EPISODE = 300   # should match rollout

TENSORBOARD_LOG_DIR = Path("logs/servos_task3")
MODEL_SAVE_DIR = Path("models/servos_best")
MODEL_SAVE_BASENAME = "best_model"  # saved as models/servos_best/best_model.zip

ENV_ID = "Upkie-Spine-Servos"
ENV_KWARGS = dict(frequency=FREQUENCY_HZ)


class ServosRewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping for Upkie-Servos:

    - Penalize large joint velocities  -> smoother motion
    - Penalize excessive knee flexion  -> avoid knees hitting ground
    - Gently discourage crazy wheel motion
    """

    def __init__(
        self,
        env: gym.Env,
        smooth_vel_weight: float = 0.002,
        knee_penalty_weight: float = 0.05,
        wheel_smooth_weight: float = 0.001,
        knee_angle_tol: float = 0.3,  # [rad] allowed flexion before penalty
    ):
        super().__init__(env)
        self.smooth_vel_weight = smooth_vel_weight
        self.knee_penalty_weight = knee_penalty_weight
        self.wheel_smooth_weight = wheel_smooth_weight
        self.knee_angle_tol = knee_angle_tol

        # Names we care about explicitly
        self.left_knee_name = "left_knee"
        self.right_knee_name = "right_knee"
        self.wheel_names = [
            n for n in self.env.observation_space.spaces.keys()
            if "wheel" in n
        ]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def _get_joint_velocities(self, obs: dict) -> np.ndarray:
        vels = []
        for _, joint_obs in obs.items():
            vels.append(float(joint_obs["velocity"][0]))
        return np.asarray(vels, dtype=np.float32)

    def _get_knee_angles(self, obs: dict) -> np.ndarray:
        vals = []
        for name in (self.left_knee_name, self.right_knee_name):
            if name in obs:
                vals.append(float(obs[name]["position"][0]))
        if not vals:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(vals, dtype=np.float32)

    def _get_wheel_velocities(self, obs: dict) -> np.ndarray:
        vals = []
        for name in self.wheel_names:
            if name in obs:
                vals.append(float(obs[name]["velocity"][0]))
        if not vals:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(vals, dtype=np.float32)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # --- 1) Smoothness: penalize overall joint velocity ---
        joint_vels = self._get_joint_velocities(obs)
        if joint_vels.size > 0:
            smooth_penalty = -self.smooth_vel_weight * float(np.mean(joint_vels ** 2))
        else:
            smooth_penalty = 0.0

        # --- 2) Knee "touching ground": penalize big flexion ---
        # We approximate "knees near the ground" as: |angle| > knee_angle_tol
        knee_angles = self._get_knee_angles(obs)
        if knee_angles.size > 0:
            excess = np.maximum(0.0, np.abs(knee_angles) - self.knee_angle_tol)
            knee_penalty = -self.knee_penalty_weight * float(np.sum(excess ** 2))
        else:
            knee_penalty = 0.0

        # --- 3) Wheels "on ground": discourage wild wheel velocity ---
        # This doesn't literally check contact, but we try to keep wheels from
        # flailing by slightly penalizing large wheel velocities.
        wheel_vels = self._get_wheel_velocities(obs)
        if wheel_vels.size > 0:
            wheel_penalty = -self.wheel_smooth_weight * float(np.mean(wheel_vels ** 2))
        else:
            wheel_penalty = 0.0

        shaped_reward = base_reward + smooth_penalty + knee_penalty + wheel_penalty

        info = dict(info)  # so we don’t mutate inner dict
        info["base_reward"] = base_reward
        info["smooth_penalty"] = smooth_penalty
        info["knee_penalty"] = knee_penalty
        info["wheel_penalty"] = wheel_penalty

        return obs, shaped_reward, terminated, truncated, info


def make_env():
    """Factory for a single wrapped Upkie-Servos env (for SB3 VecEnv)."""

    def _init():
        # Base env
        env = gym.make(ENV_ID, **ENV_KWARGS)

        # Action wrapper: compact continuous vector -> servo dict
        env = ServoVelActionWrapper(env)

        # Reward shaping operates on dict observations
        env = ServosRewardShapingWrapper(env)

        env = FallTerminationWrapper(env, fall_pitch=1.0)

        # Flatten dict observation -> Box vector
        env = ServoObsFlattenWrapper(env)

        # Episode length limit
        env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE)

        return env

    return _init


def main() -> None:
    # Register Upkie environments with Gym
    upkie.envs.register()

    # Make sure output directories exist
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Create vectorized environment
    # ------------------------------------------------------------------
    env = make_vec_env(make_env(), n_envs=N_ENVS)

    # ------------------------------------------------------------------
    # 2. Define PPO model
    # ------------------------------------------------------------------
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

    # Optional: save checkpoints every ~50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path=str(MODEL_SAVE_DIR),
        name_prefix="checkpoint",
    )

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print(f"[SERVOS] Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    # ------------------------------------------------------------------
    # 4. Save final "best_model"
    # ------------------------------------------------------------------
    model_path = MODEL_SAVE_DIR / MODEL_SAVE_BASENAME
    model.save(str(model_path))
    print(f"[SERVOS] Training finished. Model saved to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()