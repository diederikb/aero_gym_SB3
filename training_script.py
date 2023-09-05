import argparse
import json
import gymnasium as gym
from aero_gym.wrappers.reset_options import ResetOptions
# from gymnasium.wrappers import FrameStack, FlattenObservation
import numpy as np
from stable_baselines3 import DQN, TD3, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import custom_callbacks
from trajectory_generators import *
from os.path import join, isfile, dirname
from os import listdir
from pathlib import Path
import re

parser = argparse.ArgumentParser()

parser.add_argument("root_dir", type = str, help="root directory for logging")
parser.add_argument("env", type=str, help="AeroGym environment")
parser.add_argument("algorithm", type=str, help="learning algorithm")
parser.add_argument("case_name", type=str, help="case name")
parser.add_argument("--delta_t", type=float, default=0.1, help="minimum time step of the RL environment")
parser.add_argument("--t_max", type=float, default=20, help="maximum episode time length")
parser.add_argument("--xlim", type=float, nargs='*', default=[-0.75, 2.0], help="x limits of the flow domain")
parser.add_argument("--ylim", type=float, nargs='*', default=[-0.5, 0.5], help="y limits of the flow domain")
parser.add_argument("--initialization_time", type=float, default=5.0, help="the time after impulsively starting the flow for the solution that is used to initialize the episodes")
parser.add_argument("--alpha_init", type=float, default=0.0, help="the initial angle of attack in radians (used also for the flow initialization). Note that the observed alpha is relative to this one")
parser.add_argument('--use_discretized_wake', default=False, action='store_true')
parser.add_argument('--use_discrete_actions', default=False, action='store_true')
parser.add_argument("--num_discrete_actions", type=int, default=9, help="number of discrete actions (only used when use_discrete_actions=True")
parser.add_argument("--reward_type", type=int, default=3, help="see AeroGym documentation")
parser.add_argument('--observe_wake', default=False, action='store_true')
parser.add_argument('--observe_vorticity_field', default=False, action='store_true')
parser.add_argument('--observe_previous_wake', default=False, action='store_true')
parser.add_argument('--observe_h_ddot', default=False, action='store_true')
parser.add_argument('--observe_h_dot', default=False, action='store_true')
parser.add_argument('--observe_alpha_eff', default=False, action='store_true')
parser.add_argument('--observe_previous_alpha_eff', default=False, action='store_true')
parser.add_argument('--observe_circulation', default=False, action='store_true')
parser.add_argument('--observe_previous_lift', default=False, action='store_true')
parser.add_argument('--observe_previous_lift_error', default=False, action='store_true')
parser.add_argument('--observe_previous_circulatory_pressure', default=False, action='store_true')
parser.add_argument('--observe_previous_pressure', default=False, action='store_true')
parser.add_argument("--reference_lift", type=float, default=0.0, help="reference value that the lift should match")
parser.add_argument("--lift_scale", type=float, default=0.1, help="value that observed lift is scaled by")
parser.add_argument("--lift_upper_limit", type=float, default=None, help="upper limit for the unscaled lift. The episode terminates if exceeded")
parser.add_argument("--lift_lower_limit", type=float, default=None, help="lower limit for the unscaled lift. The episode terminates if exceeded")
parser.add_argument("--alpha_upper_limit", type=float, default=60 * np.pi / 180, help="upper limit for the unscaled AOA. The episode terminates if exceeded")
parser.add_argument("--alpha_lower_limit", type=float, default=-60 * np.pi / 180, help="lower limit for the unscaled AOA. The episode terminates if exceeded")
parser.add_argument("--alpha_dot_limit", type=float, default=2, help="limit for the absolute value of angular velocity. The episode terminates if exceeded")
parser.add_argument("--alpha_ddot_scale", type=float, default=0.1, help="value that input alpha_ddot gets scaled by")
parser.add_argument("--h_ddot_scale", type=float, default=0.05, help="value that input h_ddot gets scaled by")
parser.add_argument("--vorticity_scale", type=float, default=0.05, help="value that the vorticity gets scaled by")
parser.add_argument("--pressure_sensor_positions", type=float, nargs='*', default=[], help="x-coordinate of the pressure sensors in the body reference frame")
parser.add_argument("--stacked_frames", type=int, default=1, help="number of frames used in FrameStack wrapper")
parser.add_argument("--net_arch", type=str, default=None, help="network architecture (default depends on the algorithm used)")
parser.add_argument("--h_ddot_generator", type=str, default="constant(0)", help="h_ddot generator used to generate training episodes")
parser.add_argument("--reference_lift_generator", type=str, default="constant(0)", help="reference lift generator used to generate training episodes")
parser.add_argument("--sys_reinit_commands", type=str, default=None, help="file that contains julia instructions to reinitialize the system at the beginning of every episode")
parser.add_argument("--eval_sys_reinit_commands", type=str, default=None, help="file that contains julia instructions to reinitialize the system at the beginning of every evaluation episode")
parser.add_argument("--model_restart_file", type=str, default=None, help="if specified, load this model and restart the training")
parser.add_argument("--model_restart_dir", type=str, default=None, help="if specified, load the latest model in this directory and restart the training, or, if there is no saved model, start a new one")
parser.add_argument("--stats_window_size", type=int, default=10, help="episode window size for the rollout logging averages")
parser.add_argument("--n_eval_episodes", type=int, default=10, help="number of episodes used to evaluate the policy")
parser.add_argument("--eval_freq", type=int, default=10000, help="frequency in rollout timesteps of policy evaluation")
parser.add_argument("--total_timesteps", type=int, default=7.5e5, help="total training timesteps")
args = parser.parse_args()
print(args)

# Case directory (for logging/saving)
case_dir = join(args.root_dir, args.algorithm + "_" + args.case_name)
Path(case_dir).mkdir(parents=True, exist_ok=True)

policy_kwargs = dict()
if args.net_arch is not None:
    net_arch = [int(l) for l in args.net_arch.split(',')]
    policy_kwargs["net_arch"] = net_arch

if args.observe_vorticity_field:
    policy = "MultiInputPolicy"
else:
    policy = "MlpPolicy"

if args.env == 'wagner':
    base_env = gym.make(
        'aero_gym/wagner-v0', 
        render_mode="ansi", 
        t_max=args.t_max, 
        delta_t=args.delta_t, 
        use_discretized_wake=args.use_discretized_wake,
        use_discrete_actions=args.use_discrete_actions,
        num_discrete_actions=args.num_discrete_actions,
        reward_type=args.reward_type, 
        observe_h_ddot=args.observe_h_ddot,
        observe_alpha_eff=args.observe_alpha_eff,
        observe_previous_alpha_eff=args.observe_previous_alpha_eff,
        observe_wake=args.observe_wake,
        observe_previous_wake=args.observe_previous_wake,
        observe_previous_lift=args.observe_previous_lift,
        observe_previous_circulatory_pressure=args.observe_previous_circulatory_pressure,
        observe_previous_pressure=args.observe_previous_pressure,
        pressure_sensor_positions=args.pressure_sensor_positions,
        lift_termination=True,
        alpha_ddot_scale=args.alpha_ddot_scale,
        lift_scale=args.lift_scale,
        h_ddot_scale=args.h_ddot_scale)
elif args.env == 'viscous_flow':
    base_env = gym.make(
        'aero_gym/viscous_flow-v0', 
        render_mode="ansi", 
        t_max=args.t_max, 
        delta_t=args.delta_t, 
        xlim=args.xlim,
        ylim=args.ylim,
        alpha_init=args.alpha_init,
        initialization_time=args.initialization_time,
        use_discrete_actions=args.use_discrete_actions,
        num_discrete_actions=args.num_discrete_actions,
        reward_type=args.reward_type, 
        observe_h_ddot=args.observe_h_ddot,
        observe_h_dot=args.observe_h_dot,
        observe_vorticity_field=args.observe_vorticity_field,
        observe_previous_lift=args.observe_previous_lift,
        observe_previous_lift_error=args.observe_previous_lift_error,
        observe_previous_pressure=args.observe_previous_pressure,
        pressure_sensor_positions=args.pressure_sensor_positions,
        lift_termination=True,
        lift_upper_limit=args.lift_upper_limit,
        lift_lower_limit=args.lift_lower_limit,
        alpha_upper_limit=args.alpha_upper_limit,
        alpha_lower_limit=args.alpha_lower_limit,
        alpha_dot_limit=args.alpha_dot_limit,
        alpha_ddot_scale=args.alpha_ddot_scale,
        lift_scale=args.lift_scale,
        h_ddot_scale=args.h_ddot_scale,
        vorticity_scale=args.vorticity_scale)
else:
    raise NotImplementedError("Specified AeroGym environment is not implemented.")

# Create the evaluation and training environment using a wrapper around the above environment such that the same PyJulia is used. Note that if we apply the options to only one env, they will be overwritten in the other one
training_env = ResetOptions(base_env, {"h_ddot_generator": eval(args.h_ddot_generator), "reference_lift_generator": eval(args.reference_lift_generator), "sys_reinit_commands": args.sys_reinit_commands})
eval_env = ResetOptions(base_env, {"sys_reinit_commands": args.eval_sys_reinit_commands})

# Wrapping environment in a Monitor wrapper so we can monitor the rollout episode rewards and lengths
training_env = Monitor(training_env)
eval_env = Monitor(eval_env)

# Wrapping environment in a DummyVecEnv such that we can apply a VecFrameStack wrapper (because the gymnasium FrameStack wrapper doesn't work with Dict observations
training_env = DummyVecEnv([lambda: training_env])
eval_env = DummyVecEnv([lambda: eval_env])

training_env = VecFrameStack(training_env, args.stacked_frames)
eval_env = VecFrameStack(eval_env, args.stacked_frames)

if args.observe_vorticity_field:
    training_env = VecTransposeImage(training_env)
    eval_env = VecTransposeImage(eval_env)

print("observation_space:")
print(training_env.observation_space)

# If model_restart_dir is specified, find the latest model_restart_file
if args.model_restart_dir is not None:
    # Make the directory in case it doesn't exist yet
    Path(args.model_restart_dir).mkdir(parents=True, exist_ok=True)
    # Find the saved models in the directory (in case it existed before)
    saved_models = [f for f in listdir(args.model_restart_dir) if re.search(r'_\d+_steps.zip',f)]
    if saved_models:
        latest_model_idx = np.argmax([int(re.sub(r'.*_(\d+)_steps.zip', r'\1', f)) for f in saved_models])
        args.model_restart_file = join(args.model_restart_dir, saved_models[latest_model_idx])
        print("Found restart file " + args.model_restart_file)
    else:
        args.model_restart_file = None

# If a model_restart_file was provided or found the model_restart_dir, use that one to restart the model. Otherwise, create a new model
if args.model_restart_file is not None:
    print("Restarting from previously trained model")
    # Recreate the logger from the saved model
    loggerdir = dirname(args.model_restart_file)
    logger = configure(loggerdir, ["stdout", "tensorboard"])
    # Construct the filepath for the buffer	
    replay_buffer_file = re.sub(r'(_\d+_steps).zip', r'_replay_buffer\1.pkl', args.model_restart_file)

    if args.algorithm == "DQN":
        model = DQN.load(args.model_restart_file, env=training_env)
        if isfile(replay_buffer_file):
            model.load_replay_buffer(replay_buffer_file)
        print(f"The loaded model has {model.replay_buffer.size()} transitions in its buffer")
    elif args.algorithm == "TD3":
        model = TD3.load(args.model_restart_file, env=training_env)
        if isfile(replay_buffer_file):
            model.load_replay_buffer(replay_buffer_file)
        print(f"The loaded model has {model.replay_buffer.size()} transitions in its buffer")
    elif args.algorithm == "PPO":
        model = PPO.load(args.model_restart_file, env=training_env)
    else:
        raise NotImplementedError("Specified RL algorithm is not implemented.")
else:
    print("Creating new training model")
    if args.model_restart_dir is not None:
        logger = configure(args.model_restart_dir, ["stdout", "tensorboard"])
    else:
        # Instead of letting the model make the logger, we call this SB3 function ourselves, which checks the existing directories in case_dir and assigns a unique directory for the logger
        # This way, we can use the loggerdir in the custom callbacks
        logger = configure_logger(
            verbose=True,
            tensorboard_log=case_dir,
            tb_log_name=args.algorithm)
    if args.algorithm == "DQN":
        model = DQN(
            policy,
            training_env, 
            policy_kwargs=policy_kwargs,
            stats_window_size=args.stats_window_size,
            exploration_fraction=0.25, 
            verbose=1,
            batch_size=128,
            buffer_size=1_000_000,
            learning_starts=10_000)
    elif args.algorithm == "TD3":
        model = TD3(
            policy, 
            training_env, 
            policy_kwargs=policy_kwargs,
            stats_window_size=args.stats_window_size,
            verbose=1,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
            gamma=0.98,
            action_noise=NormalActionNoise(mean=np.zeros(1,dtype=np.float32),sigma=0.1*np.ones(1,dtype=np.float32)),
            train_freq=(1,"episode"))
    elif args.algorithm == "SAC":
        model = SAC(
            policy, 
            training_env, 
            policy_kwargs=policy_kwargs,
            stats_window_size=args.stats_window_size,
            verbose=1,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
            gamma=0.98,
            train_freq=(1,"episode"))
    elif args.algorithm == "PPO":
        model = PPO(
            policy, 
            training_env, 
            policy_kwargs=policy_kwargs,
            stats_window_size=args.stats_window_size,
            verbose=1,
            n_steps=256,
            n_epochs=4,
            batch_size=256,
            learning_rate=2.5e-4,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.01)
    else:
        raise NotImplementedError("Specified RL algorithm is not implemented.")

loggerdir = logger.get_dir()
print("loggerdir = " + loggerdir)
Path(loggerdir).mkdir(parents=True, exist_ok=True)

with open(join(loggerdir,'case_args.json'), 'w') as jsonfile:
    print("writing case arguments to " + join(loggerdir, 'case_args.json'))
    json.dump(vars(args), jsonfile)
    jsonfile.close()

model.set_logger(logger)

# Set up callbacks
# figure_recorder = custom_callbacks.FigureRecorderCallback(
#         eval_env,
#         log_freq=100 # rollouts
#     )
eval_callback = EvalCallback(
        eval_env,
        log_path=loggerdir,
        best_model_save_path=loggerdir,
        eval_freq=args.eval_freq, # steps
        deterministic=True,
        render=False,
        n_eval_episodes=args.n_eval_episodes
    )
checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=loggerdir,
        name_prefix="rl_model",
        save_replay_buffer=True
    )
# callback_list = CallbackList([figure_recorder, custom_callbacks.HParamCallback(), eval_callback])
# callback_list = CallbackList([custom_callbacks.HParamCallback(), eval_callback, checkpoint_callback])
callback_list = CallbackList([eval_callback, checkpoint_callback])
model.learn(total_timesteps=int(args.total_timesteps), callback=callback_list, reset_num_timesteps=False)
print("learning finished")
