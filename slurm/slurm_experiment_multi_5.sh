#!/bin/bash
#SBATCH -J CrossQ
#SBATCH -a 0-2
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem-per-cpu=4000
#SBATCH -t 72:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090'
#SBATCH -o /home/palenicek/projects/sbx-crossq/logs/%A_%a.out.log
#SBATCH -e /home/palenicek/projects/sbx-crossq/logs/%A_%a.err.log
## Make sure to create the logs directory /home/user/Documents/projects/prog/logs, BEFORE launching the jobs.

# Setup Env
SCRIPT_PATH=$(dirname $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))
echo $SCRIPT_PATH

source $SCRIPT_PATH/conda_hook
echo "Base Conda: $(which conda)"
eval "$($(which conda) shell.bash hook)"
conda activate crossq
echo "Conda Env:  $(which conda)"

export GTIMER_DISABLE='1'
echo "GTIMER_DISABLE: $GTIMER_DISABLE"

cd $SCRIPT_PATH
echo "Working Directory:  $(pwd)"

seeds_per_task=3 #$SLURM_ARRAY_TASK_COUNT
s=$SLURM_ARRAY_TASK_ID

for i in $(seq 1 $seeds_per_task); do
    python /home/palenicek/projects/sbx-crossq/train.py \
        -algo $ALGO \
        -env $ENV \
        -seed $((s*seeds_per_task + i)) \
        -critic_activation $ACT \
        -lr $LR \
        -utd $UTD \
        -policy_delay $PI_DELAY \
        -adam_b1 $B1 \
        -crossq_style $XQ_STYLE \
        -bn $BN \
        -ln $LN \
        -n_critics $N_CRITICS \
        -n_neurons $N_NEURONS \
        -bn_mode $BN_MODE \
        -bn_momentum $BN_MOM \
        -total_timesteps $STEPS \
        -eval_qbias $EVAL_QBIAS \
        -wandb_mode 'online' &
done

wait
