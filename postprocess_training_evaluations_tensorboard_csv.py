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
    
    print(run)
    run_diritems = os.listdir(os.path.join(args.parent_dir,run))
    
    csv_files = [item for item in run_diritems if item.endswith(".csv")]
    
    if len(csv_files) == 0:
        continue
    
    # Take first csv file
    path = os.path.join(args.parent_dir,run,csv_files[0])
    data = np.loadtxt(path, skiprows=1, delimiter=',')
    
    print(path)
    
    # Only one evaluation episode
    eval_mean_reward = data[:,2]
    eval_std_reward = data[:,2]
    eval_timesteps = data[:,1]
    
    # Don't append if timesteps don't agree
    if len(timesteps_list) > 1 and len(eval_timesteps) != len(timesteps_list[0]):
        print("length eval_timesteps (" + str(len(eval_timesteps)) + ") for " + run + " does not match length timesteps_list[0] (" + str(len(timesteps_list[0])) + ")")
        break
    elif len(timesteps_list) > 1 and not np.all(np.equal(eval_timesteps,timesteps_list[0])):
        print("eval_timesteps (len=" + str(len(eval_timesteps)) + ") for " + run + " does not match timesteps_list[0] (len=" + str(len(timesteps_list[0])) + ")")
        break
    
    mean_reward_list.append(eval_mean_reward)
    std_reward_list.append(eval_std_reward)
    timesteps_list.append(eval_timesteps)

# Reshape as ndarray
mean_reward_list = np.asarray(mean_reward_list)
std_reward_list = np.asarray(std_reward_list)
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
