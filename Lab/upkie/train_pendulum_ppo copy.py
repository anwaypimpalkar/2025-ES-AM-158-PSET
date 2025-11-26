#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""
PPO training script for the Upkie pendulum environment (Task 2).

Example usage:

    # 1) Launch the simulator from another terminal:
    #       ./start_simulation.sh
    #
    # 2) Train PPO on the pendulum wrapper (saves logs/models automatically):
    #       python upkie/train_pendulum_ppo.py \
    #           --total-timesteps 500000 \
    #           --save-dir models/pendulum_task2 \
    #           --log-dir logs/pendulum_task2
"""

from __future__ import annotations

import argparse
import ast
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

import upkie.envs
from upkie.logging import disable_warnings

def linear_schedule(initial_value: float):
    """Linear schedule: returns a function mapping progress_remaining in [0, 1] to a value.

    progress_remaining = 1.0 at the beginning of training and 0.0 at the end.
    """
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining * initial_value)

    return schedule


# Cosine schedule: smoothly anneal a scalar from its initial value to 0.
def cosine_schedule(initial_value: float):
    """Cosine schedule: smoothly anneal a scalar from its initial value to 0.

    progress_remaining = 1.0 at the beginning of training and 0.0 at the end.
    """

    def schedule(progress_remaining: float) -> float:
        # Map progress_remaining in [1, 0] to a cosine decay from 1 -> 0
        # progress_remaining = 1.0  -> cosine factor 1.0
        # progress_remaining = 0.0  -> cosine factor 0.0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return float(initial_value * cosine_decay)

    return schedule

def _parse_key_value_list(pairs: List[str]) -> Dict[str, object]:
    """
    Parse a list of "key=value" strings into a dictionary.

    Values are parsed with ``ast.literal_eval`` when possible and fall back to
    strings otherwise.
    """
    env_kwargs: Dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentError(
                None,
                f"Invalid env kwarg '{pair}', expected key=value",
            )
        key, raw_value = pair.split("=", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            value = raw_value
        env_kwargs[key] = value
    return env_kwargs


def _parse_hidden_sizes(value: str) -> List[int]:
    """Parse comma-separated layer sizes, e.g. ``"128,128"``."""
    if not value:
        return []
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid hidden sizes '{value}'"
        ) from exc


def build_env(
    args: argparse.Namespace,
    *,
    n_envs: Optional[int] = None,
    seed: Optional[int] = None,
    training: bool = True,
) -> VecEnv:
    """
    Create the vectorized Upkie pendulum environment.

    Args:
        args: Parsed CLI arguments.
        n_envs: Number of parallel environments (defaults to args.n_envs).
        seed: RNG seed override.
        training: Whether the resulting environment is used for training.

    Returns:
        Vectorized Gymnasium environment compatible with SB3.
    """
    env_kwargs = dict(args.env_kwargs)
    env_kwargs.setdefault("frequency", args.frequency)
    env_kwargs.setdefault("regulate_frequency", args.regulate_frequency)
    env_kwargs.setdefault("disable_env_checker", True)

    # Determine vectorized env class
    num_envs = n_envs or args.n_envs
    vec_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv

    env = make_vec_env(
        args.env_id,
        n_envs=num_envs,
        seed=seed or args.seed,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_cls,
    )

    if args.normalize:
        env = VecNormalize(
            env,
            training=training,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )
    return env


def get_callbacks(
    args: argparse.Namespace,
    eval_env: Optional[VecEnv],
    save_dir: Path,
) -> List:
    """Set up checkpoint/eval callbacks based on CLI flags."""
    callbacks: List = []

    if args.eval_freq > 0 and eval_env is not None:
        callbacks.append(
            EvalCallback(
                eval_env,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                best_model_save_path=str(save_dir / "best"),
                log_path=str(save_dir / "eval"),
            )
        )

    if args.checkpoint_freq > 0:
        checkpoint_dir = save_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="ppo_upkie",
            )
        )

    return callbacks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO trainer for the Upkie pendulum environment (Task 2)."
    )
    parser.add_argument("--env-id", default="Upkie-Spine-Pendulum")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--frequency", type=float, default=200.0)
    parser.add_argument(
        "--regulate-frequency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable dt regulation inside the environment loop.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env-kwargs",
        nargs="*",
        default=[],
        help="Extra key=value pairs forwarded to gym.make().",
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--hidden-sizes", type=_parse_hidden_sizes, default="128,128")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--n-steps", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/task2"))
    parser.add_argument("--save-dir", type=Path, default=Path("models/task2"))
    parser.add_argument("--run-name", default="ppo_pendulum")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Load an existing PPO checkpoint (.zip) before continuing training.",
    )

    args = parser.parse_args()
    args.env_kwargs = _parse_key_value_list(args.env_kwargs)
    return args


def main() -> None:
    args = parse_args()
    upkie.envs.register()
    disable_warnings()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.run_name}_{timestamp}"

    save_dir = args.save_dir / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_env(args, training=True)
    eval_env = None
    if args.eval_freq > 0:
        eval_env = build_env(
            args,
            n_envs=1,
            seed=args.seed + 42,
            training=False,
        )
        if args.normalize and isinstance(train_env, VecNormalize):
            assert isinstance(eval_env, VecNormalize)
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms

    policy_kwargs = {}
    if args.hidden_sizes:
        policy_kwargs["net_arch"] = args.hidden_sizes

    if args.resume_from is not None:
        model = PPO.load(
            str(args.resume_from),
            env=train_env,
            device=args.device,
        )
    else:
        model = PPO(
            args.policy,
            train_env,
            learning_rate=cosine_schedule(args.learning_rate),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=cosine_schedule(args.clip_range),
            ent_coef=args.ent_coef,
            n_epochs=args.n_epochs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            device=args.device,
            verbose=1,
            target_kl=0.01,
        )

    callbacks = get_callbacks(args, eval_env, save_dir)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks if callbacks else None,
        progress_bar=args.progress_bar,
    )

    final_model_path = save_dir / "ppo_upkie_final.zip"
    model.save(str(final_model_path))
    if args.normalize and isinstance(train_env, VecNormalize):
        norm_path = save_dir / "vecnormalize.pkl"
        train_env.save(str(norm_path))

    train_env.close()
    if eval_env is not None:
        eval_env.close()

    print(
        f"Training complete. Final policy saved to {final_model_path.relative_to(Path.cwd())}"
    )


if __name__ == "__main__":
    main()
