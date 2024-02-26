#!/bin/bash
#$ -N wagner
#$ -cwd
#$ -o joblog.$JOB_ID
#$ -e joberr.$JOB_ID
#$ -j n
#$ -l gpu,RTX2080Ti,h_rt=24:00:00

# move all old job logs to oldjoblogs
source_directory=$(pwd)
destination_directory=oldjoblogs
mkdir -p "$destination_directory"
find "$source_directory" -type f -name "joblog.*" -not -name "joblog.$JOB_ID" -exec mv {} "$destination_directory" \;
find "$source_directory" -type f -name "joberr.*" -not -name "joberr.$JOB_ID" -exec mv {} "$destination_directory" \;

# load modules and activate conda environment
. /u/local/Modules/default/init/modules.sh

module load gcc/11.3.0
module load julia/1.9
module load python
module load cuda
export NSLOTS=12
source ~/unsteady_aero_RL/gymnasium_28/env_gymnasium_28/bin/activate

# logdir=logs_15_point_force
# logdir=logs_17_vortex_shedding
logdir=logs_wagner
mkdir -p $logdir
# number of times each case is run
num_runs=2

case_file=case_parameters_files/wagner_steps_ramps.json
results_dir=/u/home/b/beckers/project-sofia/unsteady_aero_RL/results/wagner_impulsive_training_impulsive_evaluation_reward_3

declare -a arguments_list=(
    # "wagner TD3 full_wake_1 --t_max=20 --delta_t=0.1 --observe_previous_lift --observe_previous_wake --observe_previous_alpha_eff --stacked_frames 1 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 full_wake_2 --t_max=20 --delta_t=0.1 --observe_previous_lift --observe_previous_wake --observe_previous_alpha_eff --stacked_frames 2 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 no_wake_info_1 --t_max=20 --delta_t=0.1 --observe_previous_lift --stacked_frames 1 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 no_wake_info_2 --t_max=20 --delta_t=0.1 --observe_previous_lift --stacked_frames 2 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 pressure_2_1 --t_max=20 --delta_t=0.1 --observe_previous_lift --observe_previous_pressure --pressure_sensor_positions 0.0 0.25 --stacked_frames 1 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "wagner TD3 pressure_2_2 --t_max=20 --delta_t=0.1 --observe_previous_lift --observe_previous_pressure --pressure_sensor_positions 0.0 0.25 --stacked_frames 2 --lift_scale=0.01 --alpha_ddot_scale=0.1 --h_ddot_scale=0.01 --h_ddot_generator=random_steps_ramps()"
    # "${case_parameters_dir}/wagner.json --root_dir=${results_dir}/wagner_001 --case_name=no_wake_info_1 --stacked_frames=1 --observe_previous_lift"
    # "${case_parameters_dir}/wagner.json --root_dir=${results_dir}/wagner_001 --case_name=no_wake_info_2 --stacked_frames=2 --observe_previous_lift"
    # "${case_parameters_dir}/wagner.json --root_dir=${results_dir}/wagner_001 --case_name=pressure_info_1 --stacked_frames=1 --observe_previous_lift --observe_previous_pressure"
    # "${case_parameters_dir}/wagner.json --root_dir=${results_dir}/wagner_001 --case_name=pressure_info_2 --stacked_frames=2 --observe_previous_lift --observe_previous_pressure"
    # "${case_parameters_dir}/wagner.json --root_dir=${results_dir}/wagner_001 --case_name=full_wake_info_1 --stacked_frames=1 --observe_previous_lift --observe_previous_alpha_eff --observe_previous_wake"
    # "${case_parameters_dir}/wagner.json --root_dir=${results_dir}/wagner_001 --case_name=full_wake_info_2 --stacked_frames=2 --observe_previous_lift --observe_previous_alpha_eff --observe_previous_wake"
    
    # "${case_file} --root_dir=${results_dir} --case_name=only_lift_info_1_3_8_8 --stacked_frames=1 --reward_type=3 --net_arch 8 8"
    # "${case_file} --root_dir=${results_dir} --case_name=only_lift_info_1_3_16_16 --stacked_frames=1 --reward_type=3 --net_arch 16 16"
    # "${case_file} --root_dir=${results_dir} --case_name=only_lift_info_1_3_32_32 --stacked_frames=1 --reward_type=3 --net_arch 32 23"
    # "${case_file} --root_dir=${results_dir} --case_name=only_lift_info_1_3_64_64 --stacked_frames=1 --reward_type=3 --net_arch 64 64"
    # "${case_file} --root_dir=${results_dir} --case_name=only_lift_info_1_3_128_128 --stacked_frames=1 --reward_type=3 --net_arch 128 128"

    "${case_file} --root_dir=${results_dir} --case_name=no_wake_info_1_3 --stacked_frames=1 --observe_alpha --observe_alpha_dot --reward_type=3"
    "${case_file} --root_dir=${results_dir} --case_name=no_wake_info_2_3 --stacked_frames=2 --observe_alpha --observe_alpha_dot --reward_type=3"
    "${case_file} --root_dir=${results_dir} --case_name=pressure_info_1_3 --stacked_frames=1 --observe_alpha --observe_alpha_dot --observe_previous_pressure --reward_type=3"
    "${case_file} --root_dir=${results_dir} --case_name=pressure_info_2_3 --stacked_frames=2 --observe_alpha --observe_alpha_dot --observe_previous_pressure --reward_type=3"
    "${case_file} --root_dir=${results_dir} --case_name=full_wake_info_1_3 --stacked_frames=1 --observe_alpha --observe_alpha_dot --observe_previous_alpha_eff --observe_previous_wake --reward_type=3"
    "${case_file} --root_dir=${results_dir} --case_name=full_wake_info_2_3 --stacked_frames=2 --observe_alpha --observe_alpha_dot --observe_previous_alpha_eff --observe_previous_wake --reward_type=3"

    # "${case_parameters_dir}/wagner_steps_ramps.json --root_dir=${results_dir}/wagner_steps_ramps_001 --case_name=pressure_info_1 --stacked_frames=1 --observe_previous_lift --observe_previous_pressure"
    # "${case_parameters_dir}/wagner_steps_ramps.json --root_dir=${results_dir}/wagner_steps_ramps_001 --case_name=full_wake_info_1 --stacked_frames=1 --observe_previous_lift --observe_previous_alpha_eff --observe_previous_wake"

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

    # Vortex shedding
    # "--root_dir=${rootdir} --model_restart_dir=${rootdir}/TD3_pressure_7_4/TD3_1"
    # "viscous_flow TD3 pressure_7_4 --reference_lift_generator=constant(0.7) --h_ddot_generator=random_d_ramps(max_int_amplitude=0.01,max_d_amplitude=0.01) --alpha_init=0.52356 --initialization_time=10 --t_max=20 --delta_t=0.1 --ylim -0.55 0.85 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=1.2 --lift_lower_limit=0.2 --alpha_upper_limit=0.1 --alpha_lower_limit=-0.7 --alpha_dot_limit=1.8 --lift_scale=0.1 --alpha_ddot_scale=8.0 --vorticity_scale=10.0 --n_eval_episodes=1 --eval_freq=1000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_7_4/TD3_1 --total_timesteps=10100"
    # "viscous_flow TD3 pressure_3_4 --reference_lift_generator=constant(0.7) --h_ddot_generator=random_d_ramps(max_int_amplitude=0.001,max_d_amplitude=0.001) --alpha_init=0.52356 --initialization_time=10 --t_max=20 --delta_t=0.1 --ylim -0.55 0.65 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 0.0 0.3 --stacked_frames 4 --lift_upper_limit=1.2 --lift_lower_limit=0.2 --alpha_upper_limit=0.1 --alpha_lower_limit=-0.7 --alpha_dot_limit=1.8 --lift_scale=0.1 --alpha_ddot_scale=8.0 --vorticity_scale=10.0 --n_eval_episodes=1 --eval_freq=1000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_3_4/TD3_4 --total_timesteps=10100"
    # "viscous_flow TD3 no_wake_info --reference_lift_generator=constant(0.7) --h_ddot_generator=random_d_ramps(max_int_amplitude=0.001,max_d_amplitude=0.001) --alpha_init=0.52356 --initialization_time=10 --t_max=20 --delta_t=0.1 --ylim -0.55 0.65 --observe_previous_lift --observe_previous_lift_error --stacked_frames 4 --lift_upper_limit=1.2 --lift_lower_limit=0.2 --alpha_upper_limit=0.1 --alpha_lower_limit=-0.7 --alpha_dot_limit=1.8 --lift_scale=0.1 --alpha_ddot_scale=8.0 --vorticity_scale=10.0 --n_eval_episodes=1 --eval_freq=1000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_no_wake_info/TD3_4 --total_timesteps=10100"
    
    # Point forcing
    # "viscous_flow TD3 pressure_7_4 --eval_sys_reinit_commands=${resourcedir}/julia_system_with_forcing.jl --sys_reinit_commands=${resourcedir}/julia_system_with_random_forcing.jl --t_max=3 --delta_t=0.03 --xlim -1.0 1.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --stacked_frames 4 --lift_upper_limit=0.6 --lift_lower_limit=-0.6 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=1 --eval_freq=1000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_7_4/TD3_10 --total_timesteps=10100"
    # "viscous_flow TD3 pressure_3_4 --eval_sys_reinit_commands=${resourcedir}/julia_system_with_forcing.jl --sys_reinit_commands=${resourcedir}/julia_system_with_random_forcing.jl --t_max=3 --delta_t=0.03 --xlim -1.0 1.75 --observe_previous_lift --observe_previous_lift_error --observe_previous_pressure --pressure_sensor_positions -0.3 0.0 0.3 --stacked_frames 4 --lift_upper_limit=0.6 --lift_lower_limit=-0.6 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=1 --eval_freq=1000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_pressure_3_4/TD3_10 --total_timesteps=10100"
    # "viscous_flow TD3 no_wake_info_4 --eval_sys_reinit_commands=${resourcedir}/julia_system_with_forcing.jl --sys_reinit_commands=${resourcedir}/julia_system_with_random_forcing.jl --t_max=3 --delta_t=0.03 --xlim -1.0 1.75 --observe_previous_lift --observe_previous_lift_error --stacked_frames 4 --lift_upper_limit=0.6 --lift_lower_limit=-0.6 --lift_scale=0.1 --alpha_ddot_scale=10.0 --vorticity_scale=10.0 --n_eval_episodes=1 --eval_freq=1000 --reward_type=4 --model_restart_dir=${rootdir}/TD3_no_wake_info_4/TD3_10 --total_timesteps=10100"
)

for (( i_run = 1; i_run <= $num_runs; i_run++ ))
do
    for i_args in "${!arguments_list[@]}"; do
        arguments=${arguments_list[$i_args]}
        date_stamp=$(date +"%F-%H-%M-%S")
        case_name=$(echo "$arguments" | grep -oP -- "--case_name=\K[^ ]+")
        logfile=${case_name}_run_${i_run}_${date_stamp}.txt
        echo $logfile
        python training_script.py $arguments > $logdir/$logfile 2>&1 &
    done

    sleep 30
    # wait
    # echo "Batch $i_run out of $num_runs done"
done
wait
echo "All batches done"
