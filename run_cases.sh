#!/bin/bash
#$ -N km1_2
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

module load gcc/9.5.0
module load julia/1.9
module load python
module load cuda
export NSLOTS=12
source ~/unsteady_aero_RL/gymnasium_28/env_gymnasium_28/bin/activate

logdir=logs
# root directory for SB3 tensorboard logger and evaluator
rootdir=/u/home/b/beckers/project-sofia/unsteady_aero_RL/logs/TD3_viscous_flow_test_1
# rootdir=/u/home/b/beckers/project-sofia/unsteady_aero_RL/logs/TD3_previous_info_02_alpha_ddot_02_h_ddot_005
# number of times each case is run
num_runs=1

lift_scale=1.0
alpha_ddot_scale=1
h_ddot_scale=0.25
vorticity_scale=1.0

declare -a arguments_list=(
    # "wagner TD3 jones_1 --observe_previous_lift --observe_previous_wake --observe_previous_alpha_eff --stacked_frames 1"
    # "wagner TD3 jones_2 --observe_previous_lift --observe_previous_wake --observe_previous_alpha_eff --stacked_frames 2"
    # "wagner TD3 no_wake_info_1 --observe_previous_lift --stacked_frames 1"
    # "wagner TD3 no_wake_info_2 --observe_previous_lift --stacked_frames 2"
    # "wagner TD3 pressure_2_1 --observe_previous_lift --observe_previous_pressure --num_sensors 2 --sensor_x_min 0.0 --sensor_x_max 0.25 --stacked_frames 1"
    # "wagner TD3 pressure_2_2 --observe_previous_lift --observe_previous_pressure --num_sensors 2 --sensor_x_min 0.0 --sensor_x_max 0.25 --stacked_frames 2"
    "viscous_flow TD3 vorticity_1 --observe_lift --observe_vorticity_field --stacked_frames 1"
    # "viscous_flow TD3 vorticity_2 --observe_lift --observe_vorticity_field --stacked_frames 2"
)

for (( i_run = 1; i_run <= $num_runs; i_run++ ))
do
    for i_args in "${!arguments_list[@]}"; do
        arguments=${arguments_list[$i_args]}
        logfile=$(echo $arguments | cut -d' ' -f2)_run_${i_run}.txt
        echo $logfile
        python training_script.py $rootdir $arguments --lift_scale ${lift_scale} --alpha_ddot_scale ${alpha_ddot_scale} --h_ddot_scale ${h_ddot_scale} > $logdir/$logfile 2>&1 &
    done

    # sleep 30
    wait
    echo "Batch $i_run out of $num_runs done"
done
echo "All batches done"
