#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Task 3: Servo-based wheeled inverted pendulum wrapper
#
# This wrapper sits on top of the "Upkie-Servos" env and:
#   - converts PPO's 1D action (ground velocity) -> servo commands
#   - keeps hips/knees near neutral using position control
#   - exposes a low-dimensional observation [theta, p, theta_dot, p_dot]
#   - implements a simple stabilizing reward and termination logic
#

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from upkie.config import ROBOT_CONFIG
from upkie.logging import logger
from upkie.utils.clamp import clamp_and_warn


class UpkieServosPendulum(gym.Wrapper):
    """Servo-level wheeled inverted pendulum wrapper.

    Actions:
        a[0] = commanded ground velocity [m/s] (un-normalized, but
               [-1, 1] m/s is a sensible range).

    Observations:
        o = [theta, p, theta_dot, p_dot]

        - theta: base pitch [rad], positive when leaning forward
        - p:     ground position [m]
        - theta_dot: base pitch rate [rad/s]
        - p_dot: ground velocity [m/s]

    Reward:
        Gaussian kernels on pitch (main), position and angular velocity.

    Termination:
        - |theta| > fall_pitch
        - |p| > position_limit
        - truncated at max_time_steps of underlying env (if set)
    """

    action_space: gym.spaces.Box
    observation_space: gym.spaces.Box

    def __init__(
        self,
        env: gym.Env,
        fall_pitch: float = 1.0,
        left_wheeled: bool = True,
        max_ground_velocity: float = 1.0,
        position_limit: float = 3.0,
    ):
        super().__init__(env)

        # Unwrap underlying Gymnasium wrappers (OrderEnforcing, TimeLimit, etc.)
        # to access the base Upkie env that actually exposes attributes like
        # `get_neutral_action`, `model`, `dt`, etc.
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        self._base_env = base_env

        # Use the base env frequency if available; otherwise infer it from dt
        if getattr(self._base_env, "frequency", None) is not None:
            self.frequency = float(self._base_env.frequency)
        else:
            # Fallback: compute frequency from time step
            self.frequency = 1.0 / float(self._base_env.dt)

        # Control time step comes from the underlying Upkie env
        self.dt = float(self._base_env.dt)

        # --- Spaces ---------------------------------------------------------
        MAX_BASE_PITCH = np.pi
        MAX_GROUND_POSITION = float("inf")
        MAX_BASE_ANGULAR_VELOCITY = 1000.0  # rad/s

        obs_limit = np.array(
            [
                MAX_BASE_PITCH,
                MAX_GROUND_POSITION,
                MAX_BASE_ANGULAR_VELOCITY,
                max_ground_velocity,
            ],
            dtype=np.float32,
        )
        act_limit = np.array([max_ground_velocity], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-obs_limit,
            high=+obs_limit,
            shape=obs_limit.shape,
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=-act_limit,
            high=+act_limit,
            shape=act_limit.shape,
            dtype=np.float32,
        )

        # --- Instance attributes -------------------------------------------
        # Template servo action (moteus-like dict for all joints)
        self._servo_action: Dict[str, Dict[str, float]] = self._base_env.get_neutral_action()

        # Neutral leg positions will be measured on reset
        self._neutral_leg_positions: Dict[str, float] = {}

        self.fall_pitch = fall_pitch
        self.left_wheeled = left_wheeled
        self.max_ground_velocity = max_ground_velocity
        self.position_limit = position_limit

        # Allow setting this from outside; default to 1000 like Task 2
        if getattr(self._base_env, "max_time_steps", None) is None:
            self._base_env.max_time_steps = 1000

        self.time_step = 0

    # ------------------------------------------------------------------ #
    # Helper: extract 4D observation from spine observation
    # ------------------------------------------------------------------ #
    def _extract_obs(self, spine_observation: dict) -> np.ndarray:
        base_orientation = spine_observation["base_orientation"]
        pitch = base_orientation["pitch"]
        theta_dot = base_orientation["angular_velocity"][1]

        odom = spine_observation["wheel_odometry"]
        p = odom["position"]
        p_dot = odom["velocity"]

        obs = np.empty(4, dtype=np.float32)
        obs[0] = pitch
        obs[1] = p
        obs[2] = theta_dot
        obs[3] = p_dot
        return obs

    # ------------------------------------------------------------------ #
    # Helper: leg actions (keep hips/knees near neutral configuration)
    # ------------------------------------------------------------------ #
    def _update_leg_actions(self, spine_observation: dict) -> None:
        """Set hip/knee positions to neutral and zero velocity."""

        if not self._neutral_leg_positions:
            # On first call (after reset), record neutral leg positions
            for joint in self._base_env.model.upper_leg_joints:
                pos = spine_observation["servo"][joint.name]["position"]
                self._neutral_leg_positions[joint.name] = pos

        for joint in self._base_env.model.upper_leg_joints:
            name = joint.name
            servo = self._servo_action[name]

            # Position control towards neutral
            servo["position"] = self._neutral_leg_positions[name]
            servo["velocity"] = 0.0
            # Keep PID scales and torque limits as provided by neutral action

    # ------------------------------------------------------------------ #
    # Helper: wheel actions from ground velocity command
    # ------------------------------------------------------------------ #
    def _update_wheel_actions(self, ground_velocity: float) -> None:
        """Convert ground velocity [m/s] to left/right wheel servo commands."""

        ground_velocity = clamp_and_warn(
            ground_velocity,
            self.action_space.low[0],
            self.action_space.high[0],
            label="ground_velocity",
        )

        wheel_velocity = ground_velocity / ROBOT_CONFIG["wheel_radius"]
        left_sign = 1.0 if self.left_wheeled else -1.0
        left_wheel_velocity = left_sign * wheel_velocity
        right_wheel_velocity = -left_wheel_velocity

        for name, vel in [("left_wheel", left_wheel_velocity),
                          ("right_wheel", right_wheel_velocity)]:
            servo = self._servo_action[name]
            servo["position"] = math.nan
            servo["velocity"] = vel
            # ff torque left as template; kp/kd from neutral action

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        self.time_step = 0
        self._neutral_leg_positions.clear()

        obs_raw, info = self.env.reset(seed=seed, options=options)
        # obs_raw is a per-servo dict; we use the spine observation instead
        spine_obs = info["spine_observation"]

        # Initialize leg actions (record neutral)
        self._update_leg_actions(spine_obs)

        obs = self._extract_obs(spine_obs)
        return obs, info

    # ------------------------------------------------------------------ #
    # Fall / termination checks
    # ------------------------------------------------------------------ #
    def _detect_fall(self, spine_observation: dict) -> bool:
        pitch = spine_observation["base_orientation"]["pitch"]
        if abs(pitch) > self.fall_pitch:
            logger.warning(
                "Fall detected (pitch=%.2f rad, fall_pitch=%.2f rad)",
                abs(pitch),
                self.fall_pitch,
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 1) Update servo action dict from PPO action
        ground_velocity_cmd = float(action[0])
        self._update_wheel_actions(ground_velocity_cmd)

        # Note: _update_leg_actions needs current spine observation, which we
        # only get *after* stepping. So we rely on neutral leg positions
        # initialized at reset and updated each step with latest spine obs.

        # 2) Step underlying env with servo dict
        obs_raw, reward_env, terminated, truncated, info = self.env.step(
            self._servo_action
        )
        spine_obs = info["spine_observation"]

        # Make sure leg actions track neutral around current state
        self._update_leg_actions(spine_obs)

        observation = self._extract_obs(spine_obs)
        pitch = observation[0]
        position = observation[1]
        theta_dot = observation[2]

        # 3) Additional termination conditions
        if self._detect_fall(spine_obs):
            terminated = True

        if abs(position) > self.position_limit:
            logger.warning("Position limit reached (|p|=%.2f m)", position)
            terminated = True

        self.time_step += 1
        if getattr(self._base_env, "max_time_steps", None) is not None:
            if self.time_step >= self._base_env.max_time_steps:
                truncated = True

        # 4) Reward shaping (Gaussian kernels)
        reward_pitch = np.exp(-15.0 * pitch**2)          # main objective
        reward_position = np.exp(-1.0 * position**2)     # stay near origin
        reward_velocity = np.exp(-0.1 * theta_dot**2)    # smooth base motion

        reward = (
            1.0 * reward_pitch
            + 0.1 * reward_position
            + 0.1 * reward_velocity
        )

        if terminated:
            # Optional: you can make this slightly negative if you want
            reward = 0.0

        return observation, float(reward), bool(terminated), bool(truncated), info