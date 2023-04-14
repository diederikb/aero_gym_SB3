#!/bin/bash
#$ -N DQN
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
# number of times each case is run
num_runs=1
max_frames=1

declare -a arguments_list=(
    "DQN jones aero_gym/wagner_jones-v0 --observe_previous_lift --observe_wake"
    "DQN no_wake_info aero_gym/wagner-v0 --observe_previous_lift"
    "DQN circulation aero_gym/wagner-v0 --observe_previous_lift --observe_circulation"
    "DQN pressure_1 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 1"
    "DQN pressure_2 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 2"
    "DQN pressure_4 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 4"
    "DQN pressure_8 aero_gym/wagner-v0 --observe_previous_lift --observe_pressure --num_sensors 8"
)

for (( i_run = 1; i_run <= $num_runs; i_run++ ))
do
    for (( i_frames = 1; i_frames <= $max_frames; i_frames++ ))
    do
        for i_args in "${!arguments_list[@]}"; do
            arguments=${arguments_list[$i_args]}
            logfile=$(echo $arguments | cut -d' ' -f2)_${i_frames}.txt
            echo $logfile
            python training_script.py $arguments --stacked_frames $i_frames > $logdir/$logfile 2>&1 &
        done
    done

    wait
    echo "Batch $i_run out of $num_runs done"
done
echo "All batches done"
