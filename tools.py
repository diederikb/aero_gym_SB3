# tools.py

import gymnasium as gym
import aero_gym
from gymnasium.wrappers.render_collection import RenderCollection
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
import copy
import parsers

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

    if "observe_vorticity_field" in env_kwargs:
        if env_kwargs["observe_vorticity_field"]:
            env = VecTransposeImage(env)

    return env

class Agent:
    
    def __init__(self, name, policy, extra_cli_args=[], general_input_as_dict={}, env_input_as_dict={}):
        self.name = name
        self.policy = policy
        self.extra_cli_args = extra_cli_args
        self.general_input_as_dict = general_input_as_dict
        self.env_input_as_dict = env_input_as_dict
        self.infos = []
        self.rewards = []
        self.renders = []
        self.renders_in_progress = []

    def collect_infos(self, locals, globals):
        infos = locals["infos"]
        dones = locals["dones"]
        
        for i in range(len(dones)):
            if dones[i]:
                self.infos.append(infos[i])
                self.rewards.append(infos[i]["episode"]["r"])

    def collect_infos_and_renders(self, locals, globals):
        infos = locals["infos"]
        dones = locals["dones"]
        env = locals["env"]
        
        for i in range(len(dones)):
            if dones[i]:
                self.infos.append(infos[i])
                self.renders.append(self.renders_in_progress[i])
                self.rewards.append(infos[i]["episode"]["r"])
            else:
                if i > len(self.renders_in_progress) - 1:
                    self.renders_in_progress.append(env.unwrapped.envs[i].render())
                else:
                    self.renders_in_progress[i] = env.unwrapped.envs[i].render()

    def create_env(self, case, render):
        general_input_as_dict = copy.deepcopy(self.general_input_as_dict)
        env_input_as_dict = copy.deepcopy(self.env_input_as_dict)
        general_input_as_dict.update(case.general_input_as_dict)
        env_input_as_dict.update(case.env_input_as_dict)
        args = ["--input_file", case.input_file, *self.extra_cli_args]
        self.parsed_input_dict, self.raw_input_dict = parsers.parse_eval_args(args, general_input_as_dict, env_input_as_dict)
        
        env = create_wrapped_aerogym_env(
            case.env_name, 
            self.parsed_input_dict["env_kwargs"], 
            self.parsed_input_dict["stacked_frames"],
            collect_renders = render
        )
        
        self.infos = []
        self.renders = []

        return env

    def reset(self):
        self.infos = []
        self.rewards = []
        self.renders = []
        self.renders_in_progress = []

class Case:

    def __init__(
        self,
        name,
        env_name,
        input_file,
        agents,
        n_eval_episodes = 1,
        h_ddot_generator = None,
        reference_lift_generator = None,
        sys_reinit_commands = None,
        general_input_as_dict={},
        env_input_as_dict={}
    ):
        self.name = name
        self.env_name = env_name
        self.agents = agents
        self.input_file = input_file
        self.n_eval_episodes = n_eval_episodes
        self.h_ddot_generator = h_ddot_generator
        self.reference_lift_generator = reference_lift_generator
        self.sys_reinit_commands = sys_reinit_commands
        self.general_input_as_dict = general_input_as_dict
        self.env_input_as_dict = env_input_as_dict
