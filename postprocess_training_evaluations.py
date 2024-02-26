import numpy as np
import os
import argparse

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

parser = argparse.ArgumentParser()
parser.add_argument("parent_dir", type=str, help="parent directory of cases")
parser.add_argument("--archive_prefix", type=str, help="prefix for npz archive file name", default="")
parser.add_argument("--archive_dir", type=str, help="directory within parent_dir/run where evaluations are saved", default="archive")
args = parser.parse_args()

diritems = os.listdir(args.parent_dir)

mean_reward_list = []
std_reward_list = []
mean_ep_length_list = []
std_ep_length_list = []
timesteps_list = []

min_length = float('inf')  # Initialize with positive infinity
keys = ['timesteps', 'results', 'ep_lengths']

for run in [item for item in diritems if os.path.isdir(os.path.join(args.parent_dir, item))]:
    archives_path = os.path.join(args.parent_dir, run, args.archive_dir)
    print(archives_path)
    merged_data = dict([(key, []) for key in keys])
    encountered_timesteps = set()

    # Continue to next iteration if no data is available
    if not os.path.isdir(archives_path):
        print(f"no evaluations available for {run}")
        continue

    for archive in os.listdir(archives_path):
        if archive.startswith(args.archive_prefix):
            archive_path = os.path.join(args.parent_dir, run, args.archive_dir, archive)
            data = np.load(archive_path)
            for key in keys:
                merged_data[key].extend(data[key])

    print("shape timesteps = " + str(np.shape(merged_data["timesteps"])))
    print("shape results = " + str(np.shape(merged_data["results"])))

    # Continue to next iteration if data is empty
    if not merged_data['timesteps']:
        print("no data available")
        continue

    # Get indices order to sort data by timesteps
    ind = np.argsort(merged_data['timesteps'])

    # Average over evaluation episodes
    if np.ndim(merged_data["results"]) > 1:
        eval_mean_reward = np.mean(merged_data["results"], 1)[ind]
        eval_std_reward = np.std(merged_data["results"], 1)[ind]
        eval_mean_ep_length = np.mean(merged_data["ep_lengths"], 1)[ind]
        eval_std_ep_length = np.std(merged_data["ep_lengths"], 1)[ind]
    else:
        eval_mean_reward = merged_data["results"][ind]
        eval_std_reward = merged_data["results"][ind]
        eval_mean_ep_length = merged_data["ep_lengths"][ind]
        eval_std_ep_length = merged_data["ep_lengths"][ind]
    eval_timesteps = np.array(merged_data["timesteps"])[ind]

    print("shape timesteps after averaging = " + str(np.shape(eval_timesteps)))
    print("shape results after averaging = " + str(np.shape(eval_mean_reward)))

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
    print("unique timesteps = " + str(len(unique_timesteps)))

# Cut lists to the length of the shortest list
mean_reward_list = [arr[:min_length] for arr in mean_reward_list]
std_reward_list = [arr[:min_length] for arr in std_reward_list]
mean_ep_length_list = [arr[:min_length] for arr in mean_ep_length_list]
std_ep_length_list = [arr[:min_length] for arr in std_ep_length_list]
timesteps_list = [arr[:min_length] for arr in timesteps_list]

# Reshape as ndarray
mean_reward_array = np.asarray(mean_reward_list)
std_reward_array = np.asarray(std_reward_list)
mean_ep_length_array = np.asarray(mean_ep_length_list)
std_ep_length_array = np.asarray(std_ep_length_list)
timesteps_array = np.asarray(timesteps_list)

print("shape timesteps all cases = "+ str(np.shape(timesteps_array)))
print("shape mean rewards all cases = "+ str(np.shape(mean_reward_array)))

# Take the mean and std over the batches (runs) of the mean reward
mean_eval_mean_reward = np.mean(mean_reward_array, 0)
std_eval_mean_reward = np.std(mean_reward_array, 0)

# Take moving average
window_size = 20
print("taking moving average with window size " + str(window_size))
mean_eval_mean_reward = moving_average(mean_eval_mean_reward, window_size)
std_eval_mean_reward = moving_average(std_eval_mean_reward, window_size)
lower_eval_mean_reward = mean_eval_mean_reward - std_eval_mean_reward
upper_eval_mean_reward = mean_eval_mean_reward + std_eval_mean_reward

case_name = os.path.basename(os.path.normpath(args.parent_dir))

np.savetxt(
    os.path.join(args.parent_dir, args.archive_prefix + case_name + "_eval_mean_reward_mean.txt"),
    np.c_[
        timesteps_array[0][window_size - 1:],
        mean_eval_mean_reward,
    ]
)
np.savetxt(
    os.path.join(args.parent_dir, args.archive_prefix + case_name + "_eval_mean_reward_mean_plus_std.txt"),
    np.c_[
        timesteps_array[0][window_size - 1:],
        upper_eval_mean_reward
    ]
)
np.savetxt(
    os.path.join(args.parent_dir, args.archive_prefix + case_name + "_eval_mean_reward_mean_minus_std.txt"),
    np.c_[
        timesteps_array[0][window_size - 1:],
        lower_eval_mean_reward,
    ]
)

print(f"Data cut to the length of the shortest array ({min_length}), corresponding to timestep {timesteps_array[-1][-1]}")
