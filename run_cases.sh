#!/bin/bash
#$ -N TD3_v4
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
rootdir=/u/home/b/beckers/project-sofia/unsteady_aero_RL/logs/TD3_partial_observability_test_v4
# number of times each case is run
num_runs=3

declare -a arguments_list=(
    # "DQN jones aero_gym/wagner_jones-v0 --observe_previous_lift --observe_wake"
    "TD3 jones_1 aero_gym/wagner_jones-v0 --observe_previous_lift --observe_wake --stacked_frames 1"
    # "TD3 jones_2 aero_gym/wagner_jones-v0 --observe_previous_lift --observe_wake --stacked_frames 2"
    # "DQN no_wake_info aero_gym/wagner-v0 --observe_previous_lift"
    "TD3 no_wake_info_1 aero_gym/wagner-v0 --observe_previous_lift --stacked_frames 1"
    # "TD3 no_wake_info_2 aero_gym/wagner-v0 --observe_previous_lift --stacked_frames 2"
    # "DQN circulation aero_gym/wagner-v0 --observe_previous_lift --observe_circulation"
    # "TD3 circulation_1 aero_gym/wagner-v0 --observe_previous_lift --observe_circulation --stacked_frames 1"
    # "TD3 circulation_10 aero_gym/wagner-v0 --observe_previous_lift --observe_circulation --stacked_frames 10"
    # "TD3 circulation_100 aero_gym/wagner-v0 --observe_previous_lift --observe_circulation --stacked_frames 100"
    # "TD3 pressure_1_1 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 1 --stacked_frames 1"
    # "TD3 pressure_1_2 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 1 --stacked_frames 2"
    "TD3 pressure_2_1 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 2 --stacked_frames 1"
    # "TD3 pressure_2_2 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 2 --stacked_frames 2"
    "TD3 pressure_10_1 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 10 --stacked_frames 1"
    # "TD3 pressure_10_2 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 10 --stacked_frames 2"
    # "TD3 pressure_1_10 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 1 --stacked_frames 10"
    # "TD3 pressure_1_100 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 1 --stacked_frames 100"
    # "DQN pressure_2 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 2"
    # "DQN pressure_4 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 4"
    # "DQN pressure_8 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 8"
    # "TD3 pressure_8_400_300 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 400,300"
    # "TD3 pressure_8_800_600 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 800,600"
    # "TD3 pressure_8_400_300_300 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 400,300,300"
    # "DQN pressure_8_64_64 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 64,64"
    # "DQN pressure_8_128_128 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 128,128"
    # "DQN pressure_8_256_256 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 8 --net_arch 256,256"
    # "TD3 pressure_10_2_fourier aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 10 --stacked_frames 2"
)

for (( i_run = 1; i_run <= $num_runs; i_run++ ))
do
    for i_args in "${!arguments_list[@]}"; do
        arguments=${arguments_list[$i_args]}
        logfile=$(echo $arguments | cut -d' ' -f2)_run_${i_run}.txt
        echo $logfile
        python training_script_steps_ramps_eval.py $rootdir $arguments > $logdir/$logfile 2>&1 &
    done

    sleep 5
    echo "Batch $i_run out of $num_runs done"
done
wait
echo "All batches done"
