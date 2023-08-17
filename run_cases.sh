#!/bin/bash
#$ -N TD3_4_pf
#$ -cwd
#$ -o joblog.$JOB_ID
#$ -e joberr.$JOB_ID
#$ -j n
#$ -l gpu,RTX2080Ti,h_rt=24:00:00

# move all old job logs to oldjoblogs
mkdir -p oldjoblogs
if test -f joblog.*; then
    echo "joblog* exists. Moving to oldjoblogs"
    ls joblog* | grep -Zxv "joblog.${JOB_ID}" | xargs mv -t oldjoblogs/
fi
if test -f joberr.*; then
    echo "joberr* exists. Moving to oldjoblogs"
    ls joberr* | grep -Zxv "joberr.${JOB_ID}" | xargs mv -t oldjoblogs/
fi

# load modules and activate conda environment
. /u/local/Modules/default/init/modules.sh

module load gcc/11.3.0
module load julia/1.9
module load python
module load cuda
export NSLOTS=12
source ~/unsteady_aero_RL/gymnasium_28/env_gymnasium_28/bin/activate

logdir=logs_14_point_force
# logdir=logs_14_vortex_shedding
resourcedir=~/unsteady_aero_RL/aero_gym_SB3
mkdir -p $logdir
# root directory for SB3 tensorboard logger and evaluator
# rootdir=/u/home/b/beckers/project-sofia/unsteady_aero_RL/logs/viscous_flow_test_14_vortex_shedding
rootdir=/u/home/b/beckers/project-sofia/unsteady_aero_RL/logs/viscous_flow_test_14_point_force
# rootdir=/u/home/b/beckers/project-sofia/unsteady_aero_RL/logs/TD3_previous_info_02_alpha_ddot_02_h_ddot_005
# number of times each case is run
num_runs=5

# lift_scale=0.1
# alpha_ddot_scale=0.05
# h_ddot_scale=0.025
# vorticity_scale=1.0

declare -a arguments_list=(
    # "wagner TD3 full_wake_1 --t_max=20 --delta_t=0.1 --observe_previous_lift --observe_previous_wake --observe_previous_alpha_eff --stacked_frames 1 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 full_wake_2 --t_max=20 --delta_t=0.1 --observe_previous_lift --observe_previous_wake --observe_previous_alpha_eff --stacked_frames 2 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 no_wake_info_1 --t_max=20 --delta_t=0.1 --observe_previous_lift --stacked_frames 1 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 no_wake_info_2 --t_max=20 --delta_t=0.1 --observe_previous_lift --stacked_frames 2 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 pressure_2_1 --t_max=20 --delta_t=0.1 --observe_previous_lift --observe_previous_pressure --pressure_sensor_positions 0.0 0.25 --stacked_frames 1 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 pressure_2_2 --t_max=20 --delta_t=0.1 --observe_previous_lift --observe_previous_pressure --pressure_sensor_positions 0.0 0.25 --stacked_frames 2 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"

    # "viscous_flow TD3 no_vorticity_no_pressure_2 --h_ddot_generator=random_steps_ramps() --t_max=20 --delta_t=0.02 --observe_previous_lift --observe_previous_lift_error --stacked_frames 2 --lift_scale=1.0 --alpha_ddot_scale=5.0 --vorticity_scale=10.0 --n_eval_episodes=10"
    # "viscous_flow TD3 pressure_4_3 --h_ddot_generator=random_d_ramps(max_d_amplitude=0.1) --t_max=20 --delta_t=0.01 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=0.4 --lift_lower_limit=-0.4 --lift_scale=0.1 --alpha_ddot_scale=10.0 --h_ddot_scale=1.0 --n_eval_episodes=10 --eval_freq=20000 --reward_type=3 --model_restart_dir=${rootdir}/TD3_pressure_4_3/TD3_1"
    # "viscous_flow TD3 pressure_4_4 --h_ddot_generator=random_d_ramps(max_int_amplitude=0.99,max_d_amplitude=1) --t_max=20 --delta_t=0.1 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.0 --lift_lower_limit=-1.0 --lift_scale=0.1 --alpha_ddot_scale=10.0 --h_ddot_scale=1.0 --n_eval_episodes=10 --eval_freq=100000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_4_4/TD3_1"
    # "viscous_flow TD3 pressure_4_4 --h_ddot_generator=random_d_ramps(max_int_amplitude=0.99,max_d_amplitude=0.1) --reference_lift_generator=random_constant(-0.5,0.5) --ylim -0.75 0.75 --t_max=20 --delta_t=0.01 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.0 --lift_lower_limit=-1.0 --lift_scale=0.1 --alpha_ddot_scale=10.0 --h_ddot_scale=1.0 --n_eval_episodes=10 --eval_freq=100000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_4_5/TD3_1"
    # "viscous_flow TD3 pressure_4_3 --reference_lift_generator=random_constant(0.3,0.6) --t_max=20 --delta_t=0.1 --ylim -0.25 0.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.0 --lift_lower_limit=-0.1 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=10 --eval_freq=20000 --reward_type=3 --model_restart_dir=${rootdir}/TD3_pressure_4_3/TD3_1"
    # "viscous_flow TD3 pressure_4_4 --reference_lift_generator=random_constant(0.3,0.6) --t_max=20 --delta_t=0.1 --ylim -0.25 0.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.0 --lift_lower_limit=-0.1 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=10 --eval_freq=10000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_4_4/TD3_1"
    # "viscous_flow TD3 pressure_4_4_256 --reference_lift_generator=random_constant(0.3,0.6) --t_max=20 --delta_t=0.1 --ylim -0.25 0.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.0 --lift_lower_limit=-0.1 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=10 --eval_freq=10000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_4_4_256/TD3_1"
    # "viscous_flow TD3 pressure_4_6_256 --reference_lift_generator=random_constant(0.3,0.6) --t_max=20 --delta_t=0.1 --ylim -0.25 0.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.0 --lift_lower_limit=-0.1 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=10 --eval_freq=10000 --reward_type=6 --model_restart_dir=${rootdir}/TD3_pressure_4_6_256/TD3_1"
    # "viscous_flow SAC pressure_4_4 --reference_lift_generator=random_constant(0.3,0.6) --t_max=20 --delta_t=0.1 --ylim -0.25 0.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.0 --lift_lower_limit=-0.1 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=10 --eval_freq=10000 --reward_type=4 --model_restart_dir=${rootdir}/SAC_pressure_4_4/SAC_1"
    # "viscous_flow PPO pressure_4_4 --reference_lift_generator=random_constant(0.3,0.6) --t_max=20 --delta_t=0.01 --ylim -0.25 0.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.0 --lift_lower_limit=-0.1 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=10 --eval_freq=40000 --reward_type=4 --model_restart_dir=${rootdir}/PPO_pressure_4_4/PPO_1"
    # "viscous_flow PPO pressure_4 --reference_lift_generator=random_constant(0.3,0.6) --t_max=20 --delta_t=0.02 --ylim -0.25 0.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.0 --lift_lower_limit=-0.1 --lift_scale=0.3 --alpha_ddot_scale=5.0 --vorticity_scale=10.0 --n_eval_episodes=10 --eval_freq=20000 --reward_type=3"
    # "viscous_flow TD3 pressure_4_4 --reference_lift_generator=constant(0.7) --h_ddot_generator=random_d_ramps(max_int_amplitude=0.001,max_d_amplitude=0.001) --alpha_init=0.52356 --initialization_time=10 --t_max=20 --delta_t=0.1 --ylim -0.55 0.65 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.2 --lift_lower_limit=0.2 --alpha_upper_limit=0.1 --alpha_lower_limit=-0.7 --alpha_dot_limit=1.8 --lift_scale=0.1 --alpha_ddot_scale=8.0 --vorticity_scale=10.0 --n_eval_episodes=1 --eval_freq=10000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_4_4/TD3_1"
    "viscous_flow TD3 pressure_4_4 --system_reinitialization_commands_file=${resourcedir}/julia_system_with_forcing.txt --t_max=3 --delta_t=0.03 --xlim -1.0 1.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=0.6 --lift_lower_limit=-0.6 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=1 --eval_freq=10000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_4_4/TD3_1"
)

for (( i_run = 1; i_run <= $num_runs; i_run++ ))
do
    for i_args in "${!arguments_list[@]}"; do
        arguments=${arguments_list[$i_args]}
        date_stamp=$(date +"%F-%H-%M-%S")
        logfile=$(echo $arguments | cut -d' ' -f2)_$(echo $arguments | cut -d' ' -f3)_run_${i_run}_${date_stamp}.txt
        echo $logfile
        python training_script.py $rootdir $arguments > $logdir/$logfile 2>&1 &
    done

    # sleep 30
    wait
    echo "Batch $i_run out of $num_runs done"
done
echo "All batches done"
