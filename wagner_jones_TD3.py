import gymnasium as gym
from gymnasium.wrappers import FrameStack, FlattenObservation
import numpy as np
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import os.path
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
import custom_callbacks
import h_ddot_generators

# Case root name (for logging/saving)
case_name = "TD3_reward_type_3_saved"
# Directory where to save the model if tensorboard is not used
saved_model_dir = "saved_models"

# Time parameters
t_max = 20
delta_t = 0.1

env = gym.make(
    'aero_gym/wagner_jones-v0', 
    render_mode="ansi", 
    t_max=t_max, 
    delta_t=delta_t, 
    continuous_actions=True,
    h_ddot_generator=h_ddot_generators.random_steps_ramps,
    reward_type=3, 
    lift_threshold=0.03,
    observe_wake=True,
    observe_h_ddot=True)
env = FlattenObservation(FrameStack(env, 3))
check_env(env)

# Environment to evaluate policy (used in figure_recoder)
eval_env = deepcopy(env)


figure_recorder = custom_callbacks.FigureRecorderCallback(eval_env,log_freq=100)
callback_list = CallbackList([figure_recorder, custom_callbacks.HParamCallback()])

model = TD3(
    "MlpPolicy", 
    env, 
    verbose=1,
    batch_size=256,
    buffer_size=300_000,
    learning_starts=10_000,
    gamma=0.98,
    action_noise=NormalActionNoise(mean=np.zeros(1,dtype=np.float32),sigma=0.1*np.ones(1,dtype=np.float32)),
    train_freq=(1,"episode"),
    tensorboard_log="/u/home/b/beckers/project-sofia/unsteady_aero_RL/logs/" + case_name)
model.learn(total_timesteps=int(3e5), callback=callback_list)
print("learning finished")

# Create datetime string to include in the name of the saved model
dt = datetime.now()
dt_str = dt.strftime("%m_%d_%y_%H_%M")

# Save model
loggerdir = model.logger.get_dir()
if loggerdir is None:
    Path(saved_model_dir).mkdir(parents=True,exist_ok=True)
    saved_model_path = os.path.join(saved_model_dir, case_name + "_" + dt_str + "_" + str(model._total_timesteps) + "_timesteps")
else:
    saved_model_path = os.path.join(loggerdir, "saved_model")
model.save(saved_model_path)
