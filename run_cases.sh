#!/bin/bash
#$ -N all_cases
#$ -cwd
#$ -o joblog.$JOB_ID
#$ -e joberr.$JOB_ID
#$ -j n
#$ -l gpu,RTX2080Ti,h_rt=24:00:00

# move all old job logs to oldjoblogs
mkdir -p oldjoblogs
ls joblog* | grep -Zxv "joblog.${JOB_ID}" | xargs mv -t oldjoblogs/
ls joberr* | grep -Zxv "joberr.${JOB_ID}" | xargs mv -t oldjoblogs/

# load modules and activate conda environment
. /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate gymnasium_28

logdir=logs
# root directory for SB3 tensorboard logger and evaluator
rootdir=/u/home/b/beckers/project-sofia/unsteady_aero_RL/logs/TD3_single_environment
# number of times each case is run
num_runs=5

declare -a arguments_list=(
    # "DQN jones --use_jones_approx --observe_previous_lift --observe_wake"
    "TD3 jones_1 --use_jones_approx --observe_previous_lift --observe_wake --stacked_frames 1"
    # "TD3 jones_2 --use_jones_approx --observe_previous_lift --observe_wake --stacked_frames 2"
    # "DQN no_wake_info --observe_previous_lift"
    "TD3 no_wake_info_1 --observe_previous_lift --stacked_frames 1"
    # "TD3 no_wake_info_2 --observe_previous_lift --stacked_frames 2"
    # "DQN circulation --observe_previous_lift --observe_circulation"
    # "TD3 circulation_1 --observe_previous_lift --observe_circulation --stacked_frames 1"
    # "TD3 circulation_10 --observe_previous_lift --observe_circulation --stacked_frames 10"
    # "TD3 circulation_100 --observe_previous_lift --observe_circulation --stacked_frames 100"
    "TD3 pressure_1_1 --observe_previous_lift --observe_pressure --num_sensors 1 --stacked_frames 1"
    # "TD3 pressure_1_2 --observe_previous_lift --observe_pressure --num_sensors 1 --stacked_frames 2"
    "TD3 pressure_2_1 --observe_previous_lift --observe_pressure --num_sensors 2 --stacked_frames 1"
    # "TD3 pressure_2_2 --observe_previous_lift --observe_pressure --num_sensors 2 --stacked_frames 2"
    # "TD3 pressure_10_1 --observe_previous_lift --observe_pressure --num_sensors 10 --stacked_frames 1"
    # "TD3 pressure_10_2 --observe_previous_lift --observe_pressure --num_sensors 10 --stacked_frames 2"
    # "TD3 pressure_1_10 --observe_previous_lift --observe_pressure --num_sensors 1 --stacked_frames 10"
    # "TD3 pressure_1_100 --observe_previous_lift --observe_pressure --num_sensors 1 --stacked_frames 100"
    # "DQN pressure_2 --observe_previous_lift --observe_pressure --num_sensors 2"
    # "DQN pressure_4 --observe_previous_lift --observe_pressure --num_sensors 4"
    # "DQN pressure_8 --observe_previous_lift --observe_pressure --num_sensors 8"
    # "TD3 pressure_8_400_300 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 400,300"
    # "TD3 pressure_8_800_600 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 800,600"
    # "TD3 pressure_8_400_300_300 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 400,300,300"
    # "DQN pressure_8_64_64 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 64,64"
    # "DQN pressure_8_128_128 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 128,128"
    # "DQN pressure_8_256_256 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 256,256"
    # "TD3 pressure_10_2_fourier --observe_previous_lift --observe_pressure --num_sensors 10 --stacked_frames 2"
)

for (( i_run = 1; i_run <= $num_runs; i_run++ ))
do
    for i_args in "${!arguments_list[@]}"; do
        arguments=${arguments_list[$i_args]}
        logfile=$(echo $arguments | cut -d' ' -f2)_run_${i_run}.txt
        echo $logfile
        python training_script.py $rootdir $arguments > $logdir/$logfile 2>&1 &
    done

    # wait
    echo "Batch $i_run out of $num_runs done"
done
wait
echo "All batches done"
