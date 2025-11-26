#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria

## \namespace upkie.envs.upkie_pendulum
## \brief Environment where Upkie behaves like a wheeled inverted pendulum.

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from upkie.config import ROBOT_CONFIG
from upkie.envs.upkie_env import UpkieEnv
from upkie.exceptions import UpkieException
from upkie.logging import logger
from upkie.utils.clamp import clamp_and_warn
from upkie.utils.filters import low_pass_filter


class UpkiePendulum(gym.Wrapper):
    r"""!
    Wrapper to make Upkie act as a wheeled inverted pendulum.

    \anchor upkie_pendulum_description

    When this wrapper is applied, Upkie keeps its legs straight and actions
    only affect wheel velocities. This way, it behaves like a <a
    href="https://scaron.info/robotics/wheeled-inverted-pendulum-model.html">wheeled
    inverted pendulum</a>.

    \note For reinforcement learning with neural-network policies: the
    observation space and action space are not normalized.

    ### Action space

    The action corresponds to the ground velocity resulting from wheel
    velocities. The action vector is simply:

    \f[
    a =\begin{bmatrix} \dot{p}^* \end{bmatrix}
    \f]

    where we denote by \f$\dot{p}^*\f$ the commanded ground velocity in m/s,
    which is internally converted into wheel velocity commands. Note that,
    while this action is not normalized, [-1, 1] m/s is a reasonable range for
    ground velocities.

    ### Observation space

    Vectorized observations have the following structure:

    \f[
    \begin{align*}
    o &= \begin{bmatrix} \theta \\ p \\ \dot{\theta} \\ \dot{p} \end{bmatrix}
    \end{align*}
    \f]

    where we denote by:

    - \f$\theta\f$ the pitch angle of the base with respect to the world
      vertical, in radians. This angle is positive when the robot leans
      forward.
    - \f$p\f$ the position of the average wheel contact point, in meters.
    - \f$\dot{\theta}\f$ the body angular velocity of the base frame along its
      lateral axis, in radians per seconds.
    - \f$\dot{p}\f$ the velocity of the average wheel contact point, in meters
      per seconds.

    As with all Upkie environments, full observations from the spine (detailed
    in \ref observations) are also available in the `info` dictionary
    returned by the reset and step functions.
    """

    ## \var action_space
    ## Action space.
    action_space: gym.spaces.Box

    ## \var env
    ## Internal \ref upkie.envs.upkie_env.UpkieEnv environment.
    env: UpkieEnv

    ## \var fall_pitch
    ## Fall detection pitch angle, in radians.
    fall_pitch: float

    ## \var left_wheeled
    ## Set to True (default) if the robot is left wheeled, that is, a positive
    ## turn of the left wheel results in forward motion. Set to False for a
    ## right-wheeled variant.
    left_wheeled: bool

    ## \var observation_space
    ## Observation space.
    observation_space: gym.spaces.Box

    def __init__(
        self,
        env: UpkieEnv,
        fall_pitch: float = 1.0,
        left_wheeled: bool = True,
        max_ground_velocity: float = 3.0,
        track_limit: float = 0.5,
        max_time_steps: Optional[int] = 5000,
    ):
        r"""!
        Initialize environment.

        \param env Upkie environment to command servomotors.
        \param fall_pitch Fall detection pitch angle, in radians.
        \param left_wheeled Set to True (default) if the robot is left wheeled,
            that is, a positive turn of the left wheel results in forward
            motion. Set to False for a right-wheeled variant.
        \param max_ground_velocity Maximum commanded ground velocity in m/s.
            The default value of 1 m/s is conservative, don't hesitate to
            increase it once you feel confident in your agent.
        \param track_limit Absolute ground displacement limit in meters.
            Episodes terminate when exceeded. Set to `None` to disable.
        \param max_time_steps Maximum number of steps per episode before
            truncation. Set to `None` to keep the episode alive until failure
            conditions are triggered.
        """
        super().__init__(env)
        if env.frequency is None:
            raise UpkieException("This environment needs a loop frequency")

        MAX_BASE_PITCH: float = np.pi
        MAX_GROUND_POSITION: float = float("inf")
        MAX_BASE_ANGULAR_VELOCITY: float = 1000.0  # rad/s
        observation_limit = np.array(
            [
                MAX_BASE_PITCH,
                MAX_GROUND_POSITION,
                MAX_BASE_ANGULAR_VELOCITY,
                max_ground_velocity,
            ],
            dtype=np.float32,
        )
        action_limit = np.array([max_ground_velocity], dtype=np.float32)

        # gymnasium.Env: observation_space
        self.observation_space = gym.spaces.Box(
            -observation_limit,
            +observation_limit,
            shape=observation_limit.shape,
            dtype=observation_limit.dtype,
        )

        # gymnasium.Env: action_space
        self.action_space = gym.spaces.Box(
            -action_limit,
            +action_limit,
            shape=(1,),
            dtype=action_limit.dtype,
        )

        # Instance attributes
        self.__leg_servo_action = env.get_neutral_action()
        self.env = env
        self.fall_pitch = fall_pitch
        self.left_wheeled = left_wheeled
        self.env.max_time_steps = max_time_steps
        self.max_ground_velocity = max_ground_velocity
        self.track_limit = track_limit
        self.time_stamp = 0
        self.pitch_rate_limit = 10.0  # rad/s used for reward shaping
        self.alive_bonus = 1.0
        self.fall_penalty = 5.0
        self.off_track_penalty = 5.0
        # LQR-style reward shaping matrices and action memory
        self.Q = np.diag([12.0, 5.0, 2.0, 0.1]).astype(np.float32)
        self.R = 0.0002
        # Set smoothness penalty to zero to allow rapid reversals of wheel direction
        self.Rd = 0.0
        # Previous and last control inputs for ground velocity (1D)
        self._prev_u = np.zeros(1, dtype=np.float32)
        self.last_action = np.zeros(1, dtype=np.float32)

    def __get_env_observation(self, spine_observation: dict) -> np.ndarray:
        r"""!
        Extract environment observation from spine observation dictionary.

        \param spine_observation Spine observation dictionary.
        \return Environment observation vector.
        """
        base_orientation = spine_observation["base_orientation"]
        pitch_base_in_world = base_orientation["pitch"]
        angular_velocity_base_in_base = base_orientation["angular_velocity"]
        ground_position = spine_observation["wheel_odometry"]["position"]
        ground_velocity = spine_observation["wheel_odometry"]["velocity"]

        obs = np.empty(4, dtype=np.float32)
        obs[0] = pitch_base_in_world
        obs[1] = ground_position
        obs[2] = angular_velocity_base_in_base[1]
        obs[3] = ground_velocity
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        r"""!
        Resets the environment and get an initial observation.

        \param seed Number used to initialize the environment’s internal random
            number generator.
        \param options Currently unused.
        \return
            - `observation`: Initial vectorized observation, i.e. an element
              of the environment's `observation_space`.
            - `info`: Dictionary with auxiliary diagnostic information. For
              Upkie this is the full observation dictionary sent by the spine.
        """
        _, info = self.env.reset(seed=seed, options=options)
        self.time_stamp = 0
        # Reset control memory each episode
        self._prev_u = np.zeros(1, dtype=np.float32)
        self.last_action = np.zeros(1, dtype=np.float32)
        spine_observation = info["spine_observation"]
        for joint in self.env.model.upper_leg_joints:
            position = spine_observation["servo"][joint.name]["position"]
            self.__leg_servo_action[joint.name]["position"] = position
        observation = self.__get_env_observation(spine_observation)
        # Small randomization around upright to encourage learning stabilizing behavior
        rng = getattr(self, "np_random", np.random)
        observation[0] += rng.uniform(-0.05, 0.05)   # theta
        observation[2] += rng.uniform(-0.2, 0.2)     # theta_dot
        return observation, info

    def __get_leg_servo_action(self) -> Dict[str, Dict[str, float]]:
        r"""!
        Get servo actions for both hip and knee joints.

        \return Servo action dictionary.
        """
        for joint in self.env.model.upper_leg_joints:
            prev_position = self.__leg_servo_action[joint.name]["position"]
            new_position = low_pass_filter(
                prev_output=prev_position,
                new_input=0.0,  # go to neutral configuration
                cutoff_period=1.0,  # in roughly one second
                dt=self.env.dt,
            )
            self.__leg_servo_action[joint.name]["position"] = new_position
        return self.__leg_servo_action

    def __get_wheel_servo_action(
        self,
        left_wheel_velocity: float,
        right_wheel_velocity: float,
    ) -> Dict[str, Dict[str, float]]:
        r"""!
        Get servo actions for wheel joints.

        \param[in] left_wheel_velocity Left-wheel velocity, in rad/s.
        \param[in] right_wheel_velocity Right-wheel velocity, in rad/s.
        \return Servo action dictionary.
        """
        servo_action = {
            "left_wheel": {
                "position": math.nan,
                "velocity": left_wheel_velocity,
            },
            "right_wheel": {
                "position": math.nan,
                "velocity": right_wheel_velocity,
            },
        }
        for joint in self.env.model.wheel_joints:
            servo_action[joint.name]["maximum_torque"] = joint.limit.effort
        return servo_action

    def __get_spine_action(self, action: np.ndarray) -> Dict[str, dict]:
        r"""!
        Convert environment action to a spine action dictionary.

        \param action Environment action.
        \return Spine action dictionary.
        """
        # Single commanded ground velocity [m/s] from the policy
        ground_vel = clamp_and_warn(
            float(action[0]),
            self.action_space.low[0],
            self.action_space.high[0],
            label="ground_velocity",
        )

        wheel_radius = ROBOT_CONFIG["wheel_radius"]
        left_wheel_sign = 1.0 if self.left_wheeled else -1.0

        # Map ground velocity to wheel angular velocities for a straight-line cart
        left_wheel_velocity = left_wheel_sign * (ground_vel / wheel_radius)
        right_wheel_velocity = -left_wheel_sign * (ground_vel / wheel_radius)

        leg_servo_action = self.__get_leg_servo_action()
        wheel_servo_action = self.__get_wheel_servo_action(
            left_wheel_velocity, right_wheel_velocity
        )
        return leg_servo_action | wheel_servo_action  # wheel comes second

    def __detect_fall(self, spine_observation: dict) -> bool:
        r"""!
        Detect a fall based on the base-to-world pitch angle.

        \param spine_observation Spine observation dictionary.
        \return True if and only if a fall is detected.

        Spine observations should have a "base_orientation" key. This requires
        the \ref upkie::cpp::observers::BaseOrientation observer in the spine's
        observer pipeline.
        """
        pitch = spine_observation["base_orientation"]["pitch"]
        if abs(pitch) > self.fall_pitch:
            logger.warning(
                "Fall detected (pitch=%.2f rad, fall_pitch=%.2f rad)",
                abs(pitch),
                self.fall_pitch,
            )
            return True
        return False

    def __compute_reward(self, observation: np.ndarray) -> Tuple[float, Dict[str, float]]:
        r"""!
        Compute shaped reward and expose intermediate terms for debugging.

        The reward is based on an LQR-style quadratic cost over the state and
        control effort, plus a small penalty on action smoothness:

            r = alive_bonus - (x^T Q x + R * u^2 + R_d * (u - u_prev)^2).

        \param observation Current environment observation.
        \return Reward value and individual penalty terms.
        """
        pitch, position, pitch_rate, ground_velocity = observation

        # State vector: [theta, p, theta_dot, p_dot]
        x = np.array([pitch, position, pitch_rate, ground_velocity], dtype=np.float32)

        # Control input: commanded ground velocities for both wheels (clipped)
        u = np.clip(
            self.last_action,
            -self.max_ground_velocity,
            self.max_ground_velocity,
        ).astype(np.float32)
        du = u - self._prev_u

        state_cost = float(x @ self.Q @ x)
        # Quadratic cost on both wheel commands and their rate of change
        effort_cost = float(self.R * float(u @ u))
        smooth_cost = float(self.Rd * float(du @ du))

        total_cost = state_cost + effort_cost + smooth_cost
        reward = self.alive_bonus - total_cost

        # Update previous action for next step
        self._prev_u = u

        penalty_terms = {
            "state": state_cost,
            "effort": effort_cost,
            "smooth": smooth_cost,
            "theta2": float(pitch * pitch),
            "p2": float(position * position),
            "thetadot2": float(pitch_rate * pitch_rate),
            "pdot2": float(ground_velocity * ground_velocity),
        }
        return reward, penalty_terms

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        r"""!
        Run one timestep of the environment's dynamics.

        When the end of the episode is reached, you are responsible for calling
        `reset()` to reset the environment's state.

        \param action Action from the agent.
        \return
            - `observation`: Observation of the environment, i.e. an element
              of its `observation_space`.
            - `reward`: Reward returned after taking the action.
            - `terminated`: Whether the agent reached a terminal state,
              which may be a good or a bad thing. When true, the user needs to
              call `reset()`.
            - `truncated`: Whether the episode is reaching max number of
              steps. This boolean can signal a premature end of the episode,
              i.e. before a terminal state is reached. When true, the user
              needs to call `reset()`.
            - `info`: Dictionary with additional information, reporting in
              particular the full observation dictionary coming from the spine.
        """
        # Store last commanded ground velocities for both wheels (for reward shaping)
        self.last_action = np.array(action, dtype=np.float32).copy()
        spine_action = self.__get_spine_action(action)
        _, reward, terminated, truncated, info = self.env.step(spine_action)
        spine_observation = info["spine_observation"]
        observation = self.__get_env_observation(spine_observation)
        termination_reason = None
        fell = self.__detect_fall(spine_observation)
        if fell:
            terminated = True
            termination_reason = "fall"
        if self.track_limit is not None and abs(observation[1]) > self.track_limit:
            terminated = True
            termination_reason = "track_limit"
        shaped_reward, penalty_terms = self.__compute_reward(observation)
        reward = shaped_reward
        if termination_reason == "fall":
            reward -= self.fall_penalty
        elif termination_reason == "track_limit":
            reward -= self.off_track_penalty

        # Time-limit truncation
        time_limit_reached = False
        if self.env.max_time_steps is not None:
            self.time_stamp += 1
            if self.time_stamp >= self.env.max_time_steps:
                truncated = True
                time_limit_reached = True

        if termination_reason is not None:
            info["termination_reason"] = termination_reason
        elif time_limit_reached:
            info["termination_reason"] = "time_limit"

        info["reward_terms"] = penalty_terms
        info["alive_bonus"] = self.alive_bonus
        return observation, reward, terminated, truncated, info
