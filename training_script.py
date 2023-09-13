import sys
from copy import deepcopy
import argparse
import json
import gymnasium as gym
from aero_gym.wrappers.reset_options import ResetOptions
import numpy as np
from stable_baselines3 import DQN, TD3, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from trajectory_generators import *
from os.path import join, isfile, dirname
from os import listdir
from pathlib import Path
import re

# Use command-line input to get the input file and to overwrite any settings in the input file
cli_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
cli_env_parser = argparse.ArgumentParser(parents=[cli_parser], add_help=False, argument_default=argparse.SUPPRESS)

cli_parser.add_argument("input_file", type=str, help="json file with case parameters")
cli_parser.add_argument("--root_dir", type = str, help="root directory for logging")
cli_parser.add_argument("--env", type=str, help="AeroGym environment")
cli_parser.add_argument("--algorithm", type=str, help="learning algorithm")
cli_parser.add_argument("--policy", type=str, help="the policy model to use")
cli_parser.add_argument("--case_name", type=str, help="case name")
cli_parser.add_argument("--total_timesteps", type=int, help="total training timesteps")
cli_parser.add_argument("--stacked_frames", type=int, help="total training timesteps")
cli_parser.add_argument("--training_sys_reinit_commands", type=str, help="file that contains julia instructions to reinitialize the system at the beginning of every training episode")
cli_parser.add_argument("--eval_sys_reinit_commands", type=str, help="file that contains julia instructions to reinitialize the system at the beginning of every evaluation episode")
cli_parser.add_argument("--model_restart_file", type=str, help="if specified, load this model and restart the training")
cli_parser.add_argument("--model_restart_dir", type=str, help="if specified, load the latest model in this directory and restart the training, or, if there is no saved model, start a new one")
cli_parser.add_argument("--h_ddot_generator", type=str, help="h_ddot generator used to generate training episodes")
cli_parser.add_argument("--reference_lift_generator", type=str, help="reference lift generator used to generate training episodes")

cli_env_parser.add_argument("--delta_t", type=float, help="minimum time step of the RL environment")
cli_env_parser.add_argument("--t_max", type=float, help="maximum episode time length")
cli_env_parser.add_argument("--xlim", type=float, nargs='*', help="x limits of the flow domain")
cli_env_parser.add_argument("--ylim", type=float, nargs='*', help="y limits of the flow domain")
cli_env_parser.add_argument("--initialization_time", type=float, help="the time after impulsively starting the flow for the solution that is used to initialize the episodes")
cli_env_parser.add_argument("--alpha_init", type=float, help="the initial angle of attack in radians (used also for the flow initialization). Note that the observed alpha is relative to this one")
cli_env_parser.add_argument('--observe_wake', action='store_true')
cli_env_parser.add_argument('--observe_vorticity_field', action='store_true')
cli_env_parser.add_argument('--observe_previous_wake', action='store_true')
cli_env_parser.add_argument('--observe_h_ddot', action='store_true')
cli_env_parser.add_argument('--observe_h_dot', action='store_true')
cli_env_parser.add_argument('--observe_alpha_eff', action='store_true')
cli_env_parser.add_argument('--observe_previous_alpha_eff', action='store_true')
cli_env_parser.add_argument('--observe_circulation', action='store_true')
cli_env_parser.add_argument('--observe_previous_lift', action='store_true')
cli_env_parser.add_argument('--observe_previous_lift_error', action='store_true')
cli_env_parser.add_argument('--observe_previous_circulatory_pressure', action='store_true')
cli_env_parser.add_argument('--observe_previous_pressure', action='store_true')
cli_env_parser.add_argument("--lift_scale", type=float, help="value that observed lift is scaled by")
cli_env_parser.add_argument("--lift_upper_limit", type=float, help="upper limit for the unscaled lift. The episode terminates if exceeded")
cli_env_parser.add_argument("--lift_lower_limit", type=float, help="lower limit for the unscaled lift. The episode terminates if exceeded")
cli_env_parser.add_argument("--alpha_upper_limit", type=float, help="upper limit for the unscaled AOA. The episode terminates if exceeded")
cli_env_parser.add_argument("--alpha_lower_limit", type=float, help="lower limit for the unscaled AOA. The episode terminates if exceeded")
cli_env_parser.add_argument("--alpha_dot_limit", type=float, help="limit for the absolute value of angular velocity. The episode terminates if exceeded")
cli_env_parser.add_argument("--alpha_ddot_scale", type=float, help="value that input alpha_ddot gets scaled by")
cli_env_parser.add_argument("--h_ddot_scale", type=float, help="value that input h_ddot gets scaled by")
cli_env_parser.add_argument("--vorticity_scale", type=float, help="value that the vorticity gets scaled by")
cli_env_parser.add_argument("--pressure_sensor_positions", type=float, nargs='*', help="x-coordinate of the pressure sensors in the body reference frame")

cli_general_args, unknown_args = cli_parser.parse_known_args()
cli_env_args = cli_env_parser.parse_args(unknown_args)

# Create dicts from arguments
cli_general_dict = vars(cli_general_args)
cli_env_dict = vars(cli_env_args)

with open(cli_general_args.input_file) as jf:
    input_dict = json.load(jf)
    input_dict.update(cli_general_dict)

input_env_dict = deepcopy(input_dict["env_kwargs"])
input_env_dict.update(cli_env_dict)
input_dict["env_kwargs"] = input_env_dict

# Generate values that should be objects, but that are saved as strings in the json file
parsed_input_dict = deepcopy(input_dict)
for k in ["train_freq", "action_noise"]:
    if k in parsed_input_dict["training_algorithm_kwargs"].keys():
        parsed_input_dict["training_algorithm_kwargs"][k] = eval(input_dict["training_algorithm_kwargs"][k])

# Create defaults for some keys if they are not present in parsed_input_dict
defaults = {
        "h_ddot_generator": "constant(0)",
        "reference_lift_generator": "constant(0)",
        "training_sys_reinit_commands": None,
        "eval_sys_reinit_commands": None,
        "model_restart_dir": None,
        "model_restart_file": None,
    }
for k, v in defaults.items():
    if k not in parsed_input_dict:
        parsed_input_dict[k] = v

print(parsed_input_dict)

# Case directory (for logging/saving)
case_dir = join(parsed_input_dict["root_dir"], parsed_input_dict["algorithm"] + "_" + parsed_input_dict["case_name"])
Path(case_dir).mkdir(parents=True, exist_ok=True)

base_env = gym.make("aero_gym/" + parsed_input_dict["env"], **parsed_input_dict["env_kwargs"])

# Create the evaluation and training environment using a wrapper around the above environment such that the same PyJulia is used. Note that if we apply the options to only one env, they will be overwritten in the other one
training_env = ResetOptions(
    base_env,
    {
        "h_ddot_generator": eval(parsed_input_dict["h_ddot_generator"]),
        "reference_lift_generator": eval(parsed_input_dict["reference_lift_generator"]),
        "sys_reinit_commands": parsed_input_dict["training_sys_reinit_commands"]
    }
)
eval_env = ResetOptions(base_env, {"sys_reinit_commands": parsed_input_dict["eval_sys_reinit_commands"]})

# Wrapping environment in a Monitor wrapper so we can monitor the rollout episode rewards and lengths
training_env = Monitor(training_env)
eval_env = Monitor(eval_env)

# Wrapping environment in a DummyVecEnv such that we can apply a VecFrameStack wrapper (because the gymnasium FrameStack wrapper doesn't work with Dict observations
training_env = DummyVecEnv([lambda: training_env])
eval_env = DummyVecEnv([lambda: eval_env])

training_env = VecFrameStack(training_env, parsed_input_dict["stacked_frames"])
eval_env = VecFrameStack(eval_env, parsed_input_dict["stacked_frames"])

if "observe_vorticity_field" in parsed_input_dict["env_kwargs"].keys():
    if parsed_input_dict["observe_vorticity_field"] == True:
        training_env = VecTransposeImage(training_env)
        eval_env = VecTransposeImage(eval_env)

print("observation_space:")
print(training_env.observation_space)

# If model_restart_dir is specified, find the latest model_restart_file
if parsed_input_dict["model_restart_dir"] is not None:
    # Make the directory in case it doesn't exist yet
    Path(parsed_input_dict["model_restart_dir"]).mkdir(parents=True, exist_ok=True)
    # Find the saved models in the directory (in case it existed before)
    saved_models = [f for f in listdir(parsed_input_dict["model_restart_dir"]) if re.search(r'_\d+_steps.zip',f)]
    if saved_models:
        latest_model_idx = np.argmax([int(re.sub(r'.*_(\d+)_steps.zip', r'\1', f)) for f in saved_models])
        parsed_input_dict["model_restart_file"] = join(parsed_input_dict["model_restart_dir"], saved_models[latest_model_idx])
        print("Found restart file " + parsed_input_dict["model_restart_file"])
    else:
        parsed_input_dict["model_restart_file"] = None

# If a model_restart_file was provided or found the model_restart_dir, use that one to restart the model. Otherwise, create a new model
if parsed_input_dict["model_restart_file"] is not None:
    print("Restarting from previously trained model")
    # Recreate the logger from the saved model
    loggerdir = dirname(parsed_input_dict["model_restart_file"])
    logger = configure(loggerdir, ["stdout", "tensorboard"])
    # Construct the filepath for the buffer	
    replay_buffer_file = re.sub(r'(_\d+_steps).zip', r'_replay_buffer\1.pkl', parsed_input_dict["model_restart_file"])

    model = getattr(sys.modules[__name__], parsed_input_dict["algorithm"]).load(parsed_input_dict["model_restart_file"], env=training_env)
    if parsed_input_dict["algorithm"] in ["DQN", "TD3"]:
        if isfile(replay_buffer_file):
            model.load_replay_buffer(replay_buffer_file)
        print(f"The loaded model has {model.replay_buffer.size()} transitions in its buffer")
else:
    print("Creating new training model")
    if parsed_input_dict["model_restart_dir"] is not None:
        logger = configure(parsed_input_dict["model_restart_dir"], ["stdout", "tensorboard"])
    else:
        # Instead of letting the model make the logger, we call this SB3 function ourselves, which checks the existing directories in case_dir and assigns a unique directory for the logger
        # This way, we can use the loggerdir in the custom callbacks
        logger = configure_logger(
            verbose=True,
            tensorboard_log=case_dir,
            tb_log_name=parsed_input_dict["algorithm"])
    model = getattr(sys.modules[__name__], parsed_input_dict["algorithm"])(
        parsed_input_dict["policy"],
        training_env, 
        **parsed_input_dict["training_algorithm_kwargs"]
    )
    # model = TD3('MlpPolicy', training_env)

loggerdir = logger.get_dir()
print("loggerdir = " + loggerdir)
Path(loggerdir).mkdir(parents=True, exist_ok=True)

with open(join(loggerdir,'case_args.json'), 'w') as jsonfile:
    print("writing case arguments to " + join(loggerdir, 'case_args.json'))
    json.dump(input_dict, jsonfile)
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
        deterministic=True,
        render=False,
        **parsed_input_dict["eval_callback_kwargs"]
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
model.learn(total_timesteps=int(parsed_input_dict["total_timesteps"]), callback=callback_list, reset_num_timesteps=False)
print("learning finished")
