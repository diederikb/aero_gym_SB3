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

logdir=logs_viscous_double_aimed_point_force_flap
mkdir -p $logdir
# number of times each case is run
num_runs=1

case_file=case_parameters_files/viscous_double_aimed_point_force_flap.json
results_dir=/Users/beckers/unsteady_aero_RL/results/viscous_double_aimed_point_force_flap

case_1_id=1
# case_6_id=6
# case_7_id=7
# case_8_id=8
# case_9_id=9
# case_10_id=10

declare -a arguments_list=(
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=no_wake_info_4"
    "--input_file=${case_file} --root_dir=${results_dir} --case_name=pressure_info_3_4 --reward_type 4 --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 --model_restart_dir=${results_dir}/TD3_pressure_info_3_4/TD3_${case_1_id}"
    # Point forcing
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=no_wake_info_4 --model_restart_dir=${results_dir}/TD3_no_wake_info_4/TD3_${case_6_id}"
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=pressure_info_7_4 --reward_type 4 --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_7_4/TD3_${case_6_id}"
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=no_wake_info_4 --model_restart_dir=${results_dir}/TD3_no_wake_info_4/TD3_${case_7_id}"
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=pressure_info_7_4 --reward_type 4 --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_7_4/TD3_${case_7_id}"
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=no_wake_info_4 --model_restart_dir=${results_dir}/TD3_no_wake_info_4/TD3_${case_8_id}"
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=pressure_info_7_4 --reward_type 4 --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_7_4/TD3_${case_8_id}"
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=no_wake_info_4 --model_restart_dir=${results_dir}/TD3_no_wake_info_4/TD3_${case_9_id}"
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=pressure_info_7_4 --reward_type 4 --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_7_4/TD3_${case_9_id}"
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=no_wake_info_4 --model_restart_dir=${results_dir}/TD3_no_wake_info_4/TD3_${case_10_id}"
    # "--input_file=${case_file} --root_dir=${results_dir} --case_name=pressure_info_7_4 --reward_type 4 --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_7_4/TD3_${case_10_id}"
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
        sleep 5
    done
    # wait
    # echo "Batch $i_run out of $num_runs done"
done
wait
echo "All batches done"
