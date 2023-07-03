import argparse
import json
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
from os.path import join
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("root_dir", type = str, help="root directory for logging")
parser.add_argument("algorithm", type=str, help="learning algorithm")
parser.add_argument("case_name", type=str, help="case name")
parser.add_argument('--use_discretized_wake', default=False, action='store_true')
parser.add_argument('--use_discrete_actions', default=False, action='store_true')
parser.add_argument("--num_discrete_actions", type=int, default=9, help="number of discrete actions (only used when use_discrete_actions=True")
parser.add_argument('--observe_wake', default=False, action='store_true')
parser.add_argument('--observe_previous_wake', default=False, action='store_true')
parser.add_argument('--observe_alpha_eff', default=False, action='store_true')
parser.add_argument('--observe_previous_alpha_eff', default=False, action='store_true')
parser.add_argument('--observe_circulation', default=False, action='store_true')
parser.add_argument('--observe_previous_lift', default=False, action='store_true')
parser.add_argument('--observe_previous_circulatory_pressure', default=False, action='store_true')
parser.add_argument('--observe_previous_pressure', default=False, action='store_true')
parser.add_argument("--lift_scale", type=float, default=0.1, help="value that observed lift is scaled by")
parser.add_argument("--alpha_ddot_scale", type=float, default=0.1, help="value that input alpha_ddot gets scaled by")
parser.add_argument("--h_ddot_scale", type=float, default=0.05, help="value that input h_ddot gets scaled by")
parser.add_argument("--sensor_x_max", type=float, default=0.5, help="max x-position of sensors (default=-0.5)")
parser.add_argument("--sensor_x_min", type=float, default=-0.5, help="max x-position of sensors (default=0.5)")
parser.add_argument("--num_sensors", type=int, default=0, help="number of pressure sensors (default=0)")
parser.add_argument('--include_sensor_end_positions', dest='include_end_sensors', default=False, action='store_true')
parser.add_argument("--stacked_frames", type=int, default=1, help="number of frames used in FrameStack wrapper (default=1, i.e. no wrapper)")
parser.add_argument("--net_arch", type=str, default=None, help="network architecture (default depends on the algorithm used)")
parser.add_argument("--training_hddot_generator", type=str, default="random_steps_ramps", help="h_ddot_generator used to generate training episodes")
parser.add_argument("--evaluation_hddot_generator", type=str, default="random_steps_ramps", help="h_ddot_generator used to evaluate policy during training")
args = parser.parse_args()
print(args)

# Case directory (for logging/saving)
case_dir = join(args.root_dir, args.algorithm + "_" + args.case_name)
Path(case_dir).mkdir(parents=True, exist_ok=True)

sensor_positions = np.linspace(
        args.sensor_x_min,
        args.sensor_x_max,
        num=args.num_sensors + (2 if not args.include_end_sensors else 0)
    )

if not args.include_end_sensors:
    sensor_positions = sensor_positions[1:-1]

print("sensor positions: " + str(sensor_positions))

if args.net_arch is None:
    policy_kwargs = dict()
else:
    net_arch = [int(l) for l in args.net_arch.split(',')]
    policy_kwargs = dict(net_arch=net_arch)

if args.training_hddot_generator == "random_steps_ramps":
    training_hddot_generator = h_ddot_generators.random_steps_ramps
elif args.training_hddot_generator == "random_fourier_series":
    training_hddot_generator = h_ddot_generators.random_fourier_series
else:
    raise NotImplementedError("Specified training h_ddot_generator doesn't exist.")

if args.evaluation_hddot_generator == "random_steps_ramps":
    evaluation_hddot_generator = h_ddot_generators.random_steps_ramps
elif args.evaluation_hddot_generator == "random_fourier_series":
    evaluation_hddot_generator = h_ddot_generators.random_fourier_series
else:
    raise NotImplementedError("Specified evaluation h_ddot_generator doesn't exist.")

# Time parameters
t_max = 20
delta_t = 0.1

env = gym.make(
    'aero_gym/wagner-v0', 
    render_mode="ansi", 
    t_max=t_max, 
    delta_t=delta_t, 
    use_discretized_wake=args.use_discretized_wake,
    use_discrete_actions=args.use_discrete_actions,
    num_discrete_actions=args.num_discrete_actions,
    h_ddot_generator=training_hddot_generator,
    reward_type=3, 
    observe_alpha_eff=args.observe_alpha_eff,
    observe_previous_alpha_eff=args.observe_previous_alpha_eff,
    observe_wake=args.observe_wake,
    observe_previous_wake=args.observe_previous_wake,
    observe_previous_lift=args.observe_previous_lift,
    observe_previous_circulatory_pressure=args.observe_previous_circulatory_pressure,
    observe_previous_pressure=args.observe_previous_pressure,
    pressure_sensor_positions=sensor_positions,
    lift_termination=True,
    alpha_ddot_scale=args.alpha_ddot_scale,
    lift_scale=args.lift_scale,
    h_ddot_scale=args.h_ddot_scale,
    observe_h_ddot=False)

if args.stacked_frames > 1:
    env = FlattenObservation(FrameStack(env, args.stacked_frames))
check_env(env)

print("observation_space.shape:")
print(env.observation_space.shape)

# Environment to evaluate policy (used in figure_recoder)
eval_env = deepcopy(env)
eval_env.reset(options={"h_ddot_generator":evaluation_hddot_generator})

# This SB3 function checks the existing directories in case_dir and assigns a unique directory for the logger
logger = configure_logger(
    verbose=True,
    tensorboard_log=case_dir,
    tb_log_name=args.algorithm)
loggerdir = logger.get_dir()
Path(loggerdir).mkdir(parents=True, exist_ok=True)

print("loggerdir = " + loggerdir)

with open(join(loggerdir,'case_args.json'), 'w') as jsonfile:
    json.dump(vars(args), jsonfile)

if args.algorithm == "DQN":
    model = DQN(
        "MlpPolicy",
        env, 
        policy_kwargs=policy_kwargs,
        exploration_fraction=0.25, 
        verbose=1,
        batch_size=128,
        buffer_size=1_000_000,
        learning_starts=10000)
elif args.algorithm == "TD3":
    model = TD3(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=10_000,
        gamma=0.98,
        action_noise=NormalActionNoise(mean=np.zeros(1,dtype=np.float32),sigma=0.1*np.ones(1,dtype=np.float32)),
        train_freq=(1,"episode"))
else:
    raise NotImplementedError("Specified RL algorithm is not implemented.")


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
model.learn(total_timesteps=int(7.5e5), callback=callback_list)
print("learning finished")
