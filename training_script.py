import argparse

parser = argparse.ArgumentParser()

parser.add_argument("algorithm", type=str, help="learning algorithm")
parser.add_argument("case_name", type=str, help="case name")
parser.add_argument("env_name", type=str, help="gym environment")
parser.add_argument('--observe_wake', default=False, action='store_true')
parser.add_argument('--observed_alpha_is_eff', default=False, action='store_true')
parser.add_argument('--observe_circulation', default=False, action='store_true')
parser.add_argument('--observe_previous_lift', default=False, action='store_true')
parser.add_argument('--observe_pressure', default=False, action='store_true')
parser.add_argument("--sensor_x_max", type=float, default=0.5, help="max x-position of sensors (default=-0.5)")
parser.add_argument("--sensor_x_min", type=float, default=-0.5, help="max x-position of sensors (default=0.5)")
parser.add_argument("--num_sensors", type=int, default=0, help="number of pressure sensors (default=0)")
parser.add_argument('--include_sensor_end_positions', dest='include_end_sensors', default=False, action='store_true')
parser.add_argument("--stacked_frames", type=int, default=1, help="number of frames used in FrameStack wrapper (default=1, i.e. no wrapper)")
args = parser.parse_args()
print(args)


import gymnasium as gym
from gymnasium.wrappers import FrameStack, FlattenObservation
import numpy as np
from copy import deepcopy
from stable_baselines3 import DQN, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import configure_logger
import custom_callbacks
import h_ddot_generators


sensor_positions = np.linspace(
        args.sensor_x_min,
        args.sensor_x_max,
        num=args.num_sensors + (2 if not args.include_end_sensors else 0)
    )

if not args.include_end_sensors:
    sensor_positions = sensor_positions[1:-1]


# Case root name (for logging/saving)
case_name = args.algorithm + "_partial_observability_with_lift/" + args.case_name + "_" + str(args.stacked_frames)
# Directory where to save the model if tensorboard is not used
saved_model_dir = "saved_models"

# Time parameters
t_max = 20
delta_t = 0.1

if args.env_name == 'aero_gym/wagner-v0':
    env = gym.make(
        args.env_name, 
        render_mode="ansi", 
        t_max=t_max, 
        delta_t=delta_t, 
        continuous_actions=(False if args.algorithm == "DQN" else True),
        num_discrete_actions=9,
        h_ddot_generator=h_ddot_generators.random_steps_ramps,
        reward_type=3, 
        lift_threshold=0.03,
        observed_alpha_is_eff=args.observed_alpha_is_eff,
        observe_wake=args.observe_wake,
        observe_previous_lift=args.observe_previous_lift,
        observe_body_circulation=args.observe_circulation,
        observe_pressure=args.observe_pressure,
        pressure_sensor_positions=sensor_positions,
        observe_h_ddot=False)
else:
    env = gym.make(
        args.env_name, 
        render_mode="ansi", 
        t_max=t_max, 
        delta_t=delta_t, 
        continuous_actions=(False if args.algorithm == "DQN" else True),
        num_discrete_actions=9,
        h_ddot_generator=h_ddot_generators.random_steps_ramps,
        reward_type=3, 
        lift_threshold=0.03,
        observed_alpha_is_eff=args.observed_alpha_is_eff,
        observe_wake=args.observe_wake,
        observe_previous_lift=args.observe_previous_lift,
        observe_h_ddot=False)

if args.stacked_frames > 1:
    env = FlattenObservation(FrameStack(env, args.stacked_frames))
check_env(env)

print("observation_space.shape:")
print(env.observation_space.shape)

# Environment to evaluate policy (used in figure_recoder)
eval_env = deepcopy(env)

logger = configure_logger(
    verbose=True,
    tensorboard_log="/u/home/b/beckers/project-sofia/unsteady_aero_RL/logs/" + case_name,
    tb_log_name=args.algorithm)
loggerdir = logger.get_dir()

if args.algorithm == "DQN":
    model = DQN(
        "MlpPolicy", 
        env, 
        exploration_fraction=0.1, 
        verbose=1,
        batch_size=128,
        buffer_size=20000,
        learning_starts=10000)
else:
    model = TD3(
        "MlpPolicy", 
        env, 
        verbose=1,
        batch_size=256,
        buffer_size=300_000,
        learning_starts=10_000,
        gamma=0.98,
        action_noise=NormalActionNoise(mean=np.zeros(1,dtype=np.float32),sigma=0.1*np.ones(1,dtype=np.float32)),
        train_freq=(1,"episode"))

model.set_logger(logger)

# Set up callbacks
figure_recorder = custom_callbacks.FigureRecorderCallback(
        eval_env,
        log_freq=100 # rollouts
    )
eval_callback = EvalCallback(
        eval_env,
        log_path=loggerdir,
        best_model_save_path=loggerdir,
        eval_freq=1000, # steps
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
# callback_list = CallbackList([figure_recorder, custom_callbacks.HParamCallback(), eval_callback])
callback_list = CallbackList([custom_callbacks.HParamCallback(), eval_callback])
model.learn(total_timesteps=int(5e5), callback=callback_list)
print("learning finished")
