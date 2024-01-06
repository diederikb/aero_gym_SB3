import numpy as np
import os
import argparse

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

parser = argparse.ArgumentParser()
parser.add_argument("parent_dir", type=str, help="parent directory of cases")
args = parser.parse_args()

diritems = os.listdir(args.parent_dir)

mean_reward_list = []
std_reward_list = []
mean_ep_length_list = []
std_ep_length_list = []
timesteps_list = []

min_length = float('inf')  # Initialize with positive infinity

for run in [item for item in diritems if os.path.isdir(os.path.join(args.parent_dir, item))]:
    archives_path = os.path.join(args.parent_dir, run, "previous_archives")
    merged_data = {}
    encountered_timesteps = set()

    for archive in os.listdir(archives_path):
        archive_path = os.path.join(args.parent_dir, run, "previous_archives", archive)
        data = np.load(archive_path)
        for k, v in data.items():
            if k in merged_data.keys():
                merged_data[k] = np.append(merged_data[k], v)
            else:
                merged_data[k] = v

    # Continue to next iteration if data is empty
    if not merged_data:
        print("no data available")
        continue

    # Sort data by timesteps
    ind = np.argsort(merged_data['timesteps'])
    merged_data['timesteps'] = merged_data['timesteps'][ind]
    merged_data['results'] = merged_data['results'][ind]
    merged_data['ep_lengths'] = merged_data['ep_lengths'][ind]

    print(archives_path)

    # Average over evaluation episodes
    if np.ndim(merged_data["results"]) > 1:
        eval_mean_reward = np.mean(merged_data["results"], 1)
        eval_std_reward = np.std(merged_data["results"], 1)
        eval_mean_ep_length = np.mean(merged_data["ep_lengths"], 1)
        eval_std_ep_length = np.std(merged_data["ep_lengths"], 1)
    else:
        eval_mean_reward = merged_data["results"]
        eval_std_reward = merged_data["results"]
        eval_mean_ep_length = merged_data["ep_lengths"]
        eval_std_ep_length = merged_data["ep_lengths"]
    eval_timesteps = merged_data["timesteps"]

    # Filter out multiple entries for the same timestep
    unique_timesteps = []
    unique_eval_mean_reward = []
    unique_eval_std_reward = []
    unique_eval_mean_ep_length = []
    unique_eval_std_ep_length = []

    for timestep, mean_reward, std_reward, mean_ep_length, std_ep_length in zip(
        eval_timesteps, eval_mean_reward, eval_std_reward, eval_mean_ep_length, eval_std_ep_length
    ):
        if timestep not in encountered_timesteps:
            unique_timesteps.append(timestep)
            unique_eval_mean_reward.append(mean_reward)
            unique_eval_std_reward.append(std_reward)
            unique_eval_mean_ep_length.append(mean_ep_length)
            unique_eval_std_ep_length.append(std_ep_length)
            encountered_timesteps.add(timestep)

    mean_reward_list.append(unique_eval_mean_reward)
    std_reward_list.append(unique_eval_std_reward)
    mean_ep_length_list.append(unique_eval_mean_ep_length)
    std_ep_length_list.append(unique_eval_std_ep_length)
    timesteps_list.append(unique_timesteps)

    # Update the minimum length
    min_length = min(min_length, len(unique_timesteps))
    print(unique_timesteps[-1])

# Cut arrays to the length of the shortest array
mean_reward_list = [arr[:min_length] for arr in mean_reward_list]
std_reward_list = [arr[:min_length] for arr in std_reward_list]
mean_ep_length_list = [arr[:min_length] for arr in mean_ep_length_list]
std_ep_length_list = [arr[:min_length] for arr in std_ep_length_list]
timesteps_list = [arr[:min_length] for arr in timesteps_list]

# Reshape as ndarray
mean_reward_list = np.asarray(mean_reward_list)
std_reward_list = np.asarray(std_reward_list)
mean_ep_length_list = np.asarray(mean_ep_length_list)
std_ep_length_list = np.asarray(std_ep_length_list)
timesteps_list = np.asarray(timesteps_list)

# Take the mean and std over the batches (runs) of the mean reward
mean_eval_mean_reward = np.mean(mean_reward_list, 0)
std_eval_mean_reward = np.std(mean_reward_list, 0)

# Take moving average
window_size = 20
mean_eval_mean_reward = moving_average(mean_eval_mean_reward, window_size)
std_eval_mean_reward = moving_average(std_eval_mean_reward, window_size)
lower_eval_mean_reward = mean_eval_mean_reward - std_eval_mean_reward
upper_eval_mean_reward = mean_eval_mean_reward + std_eval_mean_reward

case_name = os.path.basename(os.path.normpath(args.parent_dir))

np.savetxt(
    os.path.join(args.parent_dir, case_name + "_eval_mean_reward_mean.txt"),
    np.c_[
        timesteps_list[0][window_size - 1:],
        mean_eval_mean_reward,
    ]
)
np.savetxt(
    os.path.join(args.parent_dir, case_name + "_eval_mean_reward_mean_plus_std.txt"),
    np.c_[
        timesteps_list[0][window_size - 1:],
        upper_eval_mean_reward
    ]
)
np.savetxt(
    os.path.join(args.parent_dir, case_name + "_eval_mean_reward_mean_minus_std.txt"),
    np.c_[
        timesteps_list[0][window_size - 1:],
        lower_eval_mean_reward,
    ]
)

print(f"Data cut to the length of the shortest array ({min_length}), corresponding to timestep {timesteps_list[-1][-1]}")
