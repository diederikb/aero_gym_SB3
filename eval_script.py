import sys
import os
from pathlib import Path
import gymnasium as gym
import aero_gym
import numpy as np
from stable_baselines3 import DQN, TD3, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from parsers import parse_eval_args
from typing import List

def find_model_files(parsed_input_dict):
    files = os.listdir(parsed_input_dict["eval_dir"])

    # Filter files based on the specified pattern and eval_freq
    model_files = [filename for filename in files \
        if filename.startswith("rl_model_") \
        and filename.endswith("_steps.zip") \
        and int(filename.split("_")[2]) % parsed_input_dict["eval_freq"] == 0 \
        and int(filename.split("_")[2]) >= parsed_input_dict["start_at"]
    ]

    # Sort files based on the extracted steps
    sorted_model_files = sorted(model_files, key=lambda filename: int(filename.split("_")[2]))
    if not sorted_model_files:
        raise FileNotFoundError("no model files found in " + parsed_input_dict["eval_dir"])

    return sorted_model_files

def find_best_reward(best_mean_reward_file_path):
    try:
        # Read the existing float value from the file
        with open(best_mean_reward_file_path, 'r') as file:
            content = file.readline().strip()
            if content:
                best_mean_reward = float(content)
                print(f"Existing best mean reward: {best_mean_reward}")
            else:
                print("Best mean reward file is empty. Setting best mean reward to -inf.")
                best_mean_reward = -np.inf
    except FileNotFoundError:
        print(f"{best_mean_reward_file_path} does not exist. Setting best mean reward to -inf.")
        best_mean_reward = -np.inf
    except ValueError:
        print(f"{best_mean_reward_file_path} does not contain a valid float. Setting best mean reward to -inf.")
        best_mean_reward = -np.inf
    return best_mean_reward

parsed_input_dict, raw_input_dict = parse_eval_args()

print(parsed_input_dict)

eval_dir = Path(parsed_input_dict["eval_dir"])
case_dir = eval_dir.parent.absolute()
best_mean_reward_file_path = os.path.join(case_dir, parsed_input_dict["archive_prefix"] + "best_mean_reward.txt")

# Case directory (for logging/saving)
archive_dir = os.path.join(parsed_input_dict["eval_dir"], parsed_input_dict["archive_dir"])
Path(archive_dir).mkdir(parents=True, exist_ok=True)

if "archive_to_overwrite" in parsed_input_dict.keys():
    evaluation_path = parsed_input_dict["archive_to_overwrite"]
    print(f"Overwriting archive at {evaluation_path}", flush=True)
    data = np.load(evaluation_path)
    evaluations_results = data["results"]
    evaluations_timesteps = data["timesteps"]
    evaluations_length = data["ep_lengths"]

    # Remove data where timesteps are greater than or equal to start_at
    mask = evaluations_timesteps < parsed_input_dict.get("start_at", 0)
    evaluations_results = list([results for i, results in enumerate(evaluations_results) if mask[i]])
    evaluations_length = list([lengths for i, lengths in enumerate(evaluations_length) if mask[i]])
    evaluations_timesteps = list(evaluations_timesteps[mask])
    
else:
    # Keep increasing file_index until a unique file name has been found
    evaluation_filename_root = parsed_input_dict["archive_prefix"] + "evaluations"
    file_index = 1
    while os.path.isfile(os.path.join(archive_dir, evaluation_filename_root + "." + str(file_index) + ".npz")):
        file_index += 1
    evaluation_path = os.path.join(archive_dir, evaluation_filename_root + "." + str(file_index))
    evaluations_results: List[List[float]] = []
    evaluations_timesteps: List[int] = []
    evaluations_length: List[List[int]] = []

print(f"Evaluation file path: {evaluation_path}", flush=True)

model_files = find_model_files(parsed_input_dict)

# Create AeroGym environment
env = gym.make(
    "aero_gym/" + parsed_input_dict["env"],
    **parsed_input_dict["env_kwargs"],
)

# Wrapping environment in a Monitor wrapper so we can monitor the rollout episode rewards and lengths
env = Monitor(env)
# Wrapping environment in a DummyVecEnv such that we can apply a VecFrameStack wrapper (because the gymnasium FrameStack wrapper doesn't work with Dict observations)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, parsed_input_dict["stacked_frames"])

if "observe_vorticity_field" in parsed_input_dict["env_kwargs"].keys():
    if parsed_input_dict["env_kwargs"]["observe_vorticity_field"] == True:
        env = VecTransposeImage(env)

print("observation_space:")
print(env.observation_space)

# Loop over the sorted files
index = 0
while index < len(model_files):
    model_file = model_files[index]
    print(f"Evaluating model at {model_file}", flush=True)
    model_path_without_zip_extension = os.path.join(parsed_input_dict["eval_dir"], model_file[:-4])
    model = getattr(sys.modules[__name__], parsed_input_dict["algorithm"]).load(model_path_without_zip_extension, env=env)

    steps = int(model_file.split("_")[2])

    # Mimic SB3 EvalCallback code
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        deterministic=True,
        n_eval_episodes=parsed_input_dict["n_eval_episodes"],
        return_episode_rewards=True
    )

    assert isinstance(episode_rewards, list)
    assert isinstance(episode_lengths, list)

    evaluations_timesteps.append(steps)
    evaluations_results.append(episode_rewards)
    evaluations_length.append(episode_lengths)

    np.savez(
        evaluation_path,
        timesteps=evaluations_timesteps,
        results=evaluations_results,
        ep_lengths=evaluations_length
    )

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    print(f"{model_file} of {eval_dir} - episode reward: {mean_reward:.2f} +/- {std_reward:.2f}", flush=True)
    print(f"{model_file} of {eval_dir} - episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}", flush=True)
    print(f"Evaluation episode(s) saved in {evaluation_path}", flush=True)

    best_mean_reward = find_best_reward(best_mean_reward_file_path)
    if mean_reward > best_mean_reward:
        print("New best mean reward!", flush=True)
        model.save(os.path.join(case_dir, parsed_input_dict["archive_prefix"] + "best_model"))
        best_mean_reward = float(mean_reward)
        with open(best_mean_reward_file_path, 'w') as file:
            file.write(str(best_mean_reward))
    
    # update model_files in case new models were generated in the meantime
    model_files = find_model_files(parsed_input_dict)

    index += 1

# Run another garbage collect
if parsed_input_dict["env"] == "viscous_flow-v0":
    env.reset()
