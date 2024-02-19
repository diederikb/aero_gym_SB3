import sys
import gymnasium as gym
import aero_gym
import numpy as np
import json
from stable_baselines3 import DQN, TD3, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from parsers import parse_training_args
from tools import create_wrapped_aerogym_env
from pathlib import Path
import re
import os

parsed_input_dict, raw_input_dict = parse_training_args(sys.argv[1:])

print(parsed_input_dict)

# Case directory (for logging/saving)
case_dir = os.path.join(parsed_input_dict["root_dir"], parsed_input_dict["algorithm"] + "_" + parsed_input_dict["case_name"])
Path(case_dir).mkdir(parents=True, exist_ok=True)

env = create_wrapped_aerogym_env(parsed_input_dict["env"], parsed_input_dict["env_kwargs"], parsed_input_dict["stacked_frames"])

print("observation_space:")
print(env.observation_space)

# If model_restart_dir is specified, find the latest model_restart_file that has a matching replay buffer
if parsed_input_dict["model_restart_dir"] is not None:
    # Make the directory in case it doesn't exist yet
    Path(parsed_input_dict["model_restart_dir"]).mkdir(parents=True, exist_ok=True)
    # Find the saved models and replay buffers in the directory
    saved_buffers_dict = {int(m.group(1)): f for f in os.listdir(parsed_input_dict["model_restart_dir"]) if (m := re.search(r'_(\d+)_steps.pkl',f))}
    saved_models_dict =  {int(m.group(1)): f for f in os.listdir(parsed_input_dict["model_restart_dir"]) if (m := re.search(r'_(\d+)_steps.zip',f))}
    largest_common_step = max(set(saved_buffers_dict.keys()) & set(saved_models_dict.keys()), default=None)
    if largest_common_step:
        parsed_input_dict["model_restart_file"] = os.path.join(parsed_input_dict["model_restart_dir"], saved_models_dict[largest_common_step])
        print("Found restart file with matching replay buffer in specified restart dir (" + parsed_input_dict["model_restart_dir"] + "): " + parsed_input_dict["model_restart_file"])
    else:
        parsed_input_dict["model_restart_file"] = None
        print("Found NO restart file or restart file with matching buffer in specified restart dir (" + parsed_input_dict["model_restart_dir"] + ")")

# If a model_restart_file was provided or was found in the model_restart_dir, use that one to restart the model. Otherwise, create a new model
if parsed_input_dict["model_restart_file"] is not None:
    print("Restarting from previously trained model")
    # Recreate the logger from the saved model
    loggerdir = os.path.dirname(parsed_input_dict["model_restart_file"])
    logger = configure(loggerdir, ["stdout", "tensorboard"])
    # Construct the filepath for the buffer	
    replay_buffer_file = re.sub(r'(_\d+_steps).zip', r'_replay_buffer\1.pkl', parsed_input_dict["model_restart_file"])

    model = getattr(sys.modules[__name__], parsed_input_dict["algorithm"]).load(parsed_input_dict["model_restart_file"], env=env)
    if parsed_input_dict["algorithm"] in ["DQN", "TD3"]:
        if os.path.isfile(replay_buffer_file):
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

    policy_kwargs = {k: v for (k,v) in parsed_input_dict.items() if k in ["net_arch"]}
    model = getattr(sys.modules[__name__], parsed_input_dict["algorithm"])(
        parsed_input_dict["policy"],
        env, 
        policy_kwargs=policy_kwargs,
        **parsed_input_dict["training_algorithm_kwargs"]
    )

loggerdir = str(logger.get_dir())
print("loggerdir = " + loggerdir)
Path(loggerdir).mkdir(parents=True, exist_ok=True)

with open(os.path.join(loggerdir,'case_args.json'), 'w') as jsonfile:
    print("writing case arguments to " + os.path.join(loggerdir, 'case_args.json'))
    json.dump(raw_input_dict, jsonfile)
    jsonfile.close()

model.set_logger(logger)

# Save model every 1000 steps for evaluation purposes
checkpoint_callback = CheckpointCallback(
        save_path=loggerdir,
        name_prefix="rl_model",
        save_freq=1000,
    )

# Save model every eval_freq steps for restart purposes
checkpoint_with_buffer_callback = CheckpointCallback(
        save_path=loggerdir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        **parsed_input_dict["checkpoint_callback_kwargs"]
    )

callback_list = CallbackList([checkpoint_callback, checkpoint_with_buffer_callback])
model.learn(total_timesteps=int(parsed_input_dict["total_timesteps"]), callback=callback_list, reset_num_timesteps=False)

# Run another garbage collect
if parsed_input_dict["env"] == "viscous_flow-v0":
    env.reset()

print("learning finished")
