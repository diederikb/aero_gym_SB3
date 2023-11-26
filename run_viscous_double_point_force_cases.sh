#!/bin/bash
#$ -N double_point_force
#$ -cwd
#$ -o joblog.$JOB_ID
#$ -e joberr.$JOB_ID
#$ -j n
#$ -l gpu,V100,h_rt=24:00:00

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

logdir=logs_viscous_double_point_force
mkdir -p $logdir
# number of times each case is run
num_runs=10

case_file=case_parameters_files/viscous_double_point_force_high_amplitude.json
results_dir=/u/home/b/beckers/project-sofia/unsteady_aero_RL/results/viscous_double_point_force_high_amplitude

case_1_id=1
case_2_id=2

declare -a arguments_list=(
    # Point forcing
    # "${case_file} --root_dir=${results_dir} --case_name=no_wake_info_4 --model_restart_dir=${results_dir}/TD3_no_wake_info_4/TD3_${case_1_id}"
    # "${case_file} --root_dir=${results_dir} --case_name=pressure_info_3_4 --observe_previous_pressure --pressure_sensor_positions -0.3 0.0 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_3_4/TD3_${case_1_id}"
    "${case_file} --root_dir=${results_dir} --case_name=pressure_info_7_4 --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_7_4/TD3_${case_1_id}"
    # "${case_file} --root_dir=${results_dir} --case_name=no_wake_info_4 --model_restart_dir=${results_dir}/TD3_no_wake_info_4/TD3_${case_2_id}"
    # "${case_file} --root_dir=${results_dir} --case_name=pressure_info_3_4 --observe_previous_pressure --pressure_sensor_positions -0.3 0.0 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_3_4/TD3_${case_2_id}"
    "${case_file} --root_dir=${results_dir} --case_name=pressure_info_7_4 --observe_previous_pressure --pressure_sensor_positions -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 --model_restart_dir=${results_dir}/TD3_pressure_info_7_4/TD3_${case_2_id}"
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
    wait
    echo "Batch $i_run out of $num_runs done"
done
echo "All batches done"
