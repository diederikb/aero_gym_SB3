import numpy as np
import os
import argparse

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

parser = argparse.ArgumentParser()
parser.add_argument("parent_dir", type = str, help="parent directory of cases")
args = parser.parse_args()

diritems = os.listdir(args.parent_dir)

mean_reward_list = []
std_reward_list = []
mean_ep_length_list = []
std_ep_length_list = []
timesteps_list = []

for run in [item for item in diritems if os.path.isdir(os.path.join(args.parent_dir, item))]:
    archives_path = os.path.join(args.parent_dir,run,"previous_archives")
    merged_data = {}
    for archive in os.listdir(archives_path):
        archive_path = os.path.join(args.parent_dir,run,"previous_archives",archive)
        data = np.load(archive_path)
        for k, v in data.items():
            if k in merged_data.keys():
                merged_data[k] = np.append(merged_data[k],v)
            else:
                merged_data[k] = v
    # Continue to next iteration if data is empty
    if not merged_data:
        print("no data available")
        continue

    ind = np.argsort(merged_data['timesteps'])
    merged_data['timesteps'] = merged_data['timesteps'][ind]
    merged_data['results'] = merged_data['results'][ind]
    merged_data['ep_lengths'] = merged_data['ep_lengths'][ind]
    
    print(archives_path)
    
    # Average over evaluation episodes
    if np.ndim(merged_data["results"]) > 1:
        eval_mean_reward = np.mean(merged_data["results"],1)
        eval_std_reward = np.std(merged_data["results"],1)
        eval_mean_ep_length = np.mean(merged_data["ep_lengths"],1)
        eval_std_ep_length = np.std(merged_data["ep_lengths"],1)
    else:
        eval_mean_reward = merged_data["results"]
        eval_std_reward = merged_data["results"]
        eval_mean_ep_length = merged_data["ep_lengths"]
        eval_std_ep_length = merged_data["ep_lengths"]
    eval_timesteps = merged_data["timesteps"]
    
    # Don't append if timesteps don't agree
    if len(timesteps_list) > 1 and len(eval_timesteps) != len(timesteps_list[0]):
        print("length eval_timesteps (" + str(len(eval_timesteps)) + ") for " + run + " does not match length timesteps_list[0] (" + str(len(timesteps_list[0])) + ")")
        continue
    elif len(timesteps_list) > 1 and not np.all(np.equal(eval_timesteps,timesteps_list[0])):
        print("eval_timesteps (len=" + str(len(eval_timesteps)) + ") for " + run + " does not match timesteps_list[0] (len=" + str(len(timesteps_list[0])) + ")")
        continue
    
    mean_reward_list.append(eval_mean_reward)
    std_reward_list.append(eval_std_reward)
    mean_ep_length_list.append(eval_mean_ep_length)
    std_ep_length_list.append(eval_std_ep_length)
    timesteps_list.append(eval_timesteps)

# Reshape as ndarray
mean_reward_list = np.asarray(mean_reward_list)
std_reward_list = np.asarray(std_reward_list)
mean_ep_length_list = np.asarray(mean_ep_length_list)
std_ep_length_list = np.asarray(std_ep_length_list)
timesteps_list = np.asarray(timesteps_list)

# Take the mean and std over the batches (runs) of the mean reward
mean_eval_mean_reward = np.mean(mean_reward_list,0)
std_eval_mean_reward = np.std(mean_reward_list,0)

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
        timesteps_list[0][window_size-1:],
        mean_eval_mean_reward,
    ]
)
np.savetxt(
    os.path.join(args.parent_dir, case_name + "_eval_mean_reward_mean_plus_std.txt"),
    np.c_[
        timesteps_list[0][window_size-1:],
        upper_eval_mean_reward
    ]
)
np.savetxt(
    os.path.join(args.parent_dir, case_name + "_eval_mean_reward_mean_minus_std.txt"),
    np.c_[
        timesteps_list[0][window_size-1:],
        lower_eval_mean_reward,
    ]
)
