# tools.py

import gymnasium as gym
import aero_gym
from gymnasium.wrappers.render_collection import RenderCollection
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

def create_wrapped_aerogym_env(env_name, env_kwargs, stacked_frames, monitor_filename=None, collect_renders=False):
    env = gym.make(
        "aero_gym/" + env_name,
        **env_kwargs,
    )

    info_keywords = (
        "t_hist",
        "h_dot_hist",
        "h_ddot_hist",
        "alpha_hist",
        "alpha_dot_hist",
        "alpha_ddot_hist",
        "fy_hist",
        "reference_lift_hist",
        "t",
        "time_step",
    )
    if env_name == "viscous_flow-v0":
        info_keywords += ("solver_fy_hist", "solver_t_hist")

    if collect_renders:
        env = RenderCollection(env, pop_frames=False, reset_clean=True)

    # Wrapping environment in a Monitor wrapper so we can monitor the rollout episode rewards and lengths
    env = Monitor(env, filename=monitor_filename, info_keywords=info_keywords)
    # Wrapping environment in a DummyVecEnv such that we can apply a VecFrameStack wrapper (because the gymnasium FrameStack wrapper doesn't work with Dict observations)
    env = DummyVecEnv([lambda: env]) # type: ignore[list-item, return-value]
    env = VecFrameStack(env, stacked_frames)

    if env_kwargs["observe_vorticity_field"]:
        env = VecTransposeImage(env)

    return env
