#!/bin/bash
#$ -N double_point_force
#$ -cwd
#$ -o joblog.$JOB_ID
#$ -e joberr.$JOB_ID
#$ -j n
#$ -l gpu,V100,h_rt=24:00:00

# load modules and activate conda environment
# . /u/local/Modules/default/init/modules.sh

# module load gcc/11.3.0
# module load julia/1.9
# module load python
# module load cuda
# export NSLOTS=12
source ~/unsteady_aero_RL/.venv/bin/activate

logdir=logs_viscous_double_point_force_eval
mkdir -p $logdir

case_file=case_parameters_files/viscous_double_point_force.json
results_dir=/Users/beckers/unsteady_aero_RL/results/viscous_double_point_force
sys_reinit_commands=julia_sys_reinit_commands_files/julia_system_with_two_prescribed_force_pulses.jl
prefix="double_pulse_"

case_1_id=2
# case_2_id=2

declare -a arguments_list=(
    # Point forcing
    # "${case_file} --eval_dir=${results_dir}/TD3_no_wake_info_4/TD3_${case_1_id} --archive_prefix ${prefix} --sys_reinit_commands ${sys_reinit_commands}"
    # "${case_file} --root_dir=${results_dir} --case_name=pressure_info_3_4 --observe_previous_pressure --pressure_sensor_positions -0.3 0.0 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_3_4/TD3_${case_1_id}"
    "${case_file} --eval_dir=${results_dir}/TD3_pressure_info_7_5/TD3_${case_1_id} --start_at 194000 --archive_to_overwrite=${results_dir}/TD3_pressure_info_7_5/TD3_${case_1_id}/archive/double_pulse_evaluations.1.npz --reward_type 5 --sys_reinit_commands ${sys_reinit_commands} --archive_prefix ${prefix} --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3"
    # "${case_file} --root_dir=${results_dir} --case_name=no_wake_info_4 --model_restart_dir=${results_dir}/TD3_no_wake_info_4/TD3_${case_2_id}"
    # "${case_file} --root_dir=${results_dir} --case_name=pressure_info_3_4 --observe_previous_pressure --pressure_sensor_positions -0.3 0.0 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_3_4/TD3_${case_2_id}"
    # "${case_file} --root_dir=${results_dir} --case_name=pressure_info_7_4 --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_7_4/TD3_${case_2_id}"
)

for i_args in "${!arguments_list[@]}"; do
    arguments=${arguments_list[$i_args]}
    date_stamp=$(date +"%F-%H-%M-%S")
    case_name=$(echo "$arguments" | grep -oP -- "--case_name=\K[^ ]+")
    logfile=${date_stamp}.txt
    echo $logfile
    python eval_script.py $arguments > >(tee $logdir/$logfile) 2>&1 &
    wait
done

echo "All batches done"
