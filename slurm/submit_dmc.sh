#!/bin/bash

timesteps=1000000
q_bias=0

# learning rate 3e-4
# start learning 2500
# policy delay

# for env in 'dm_control/ball_in_cup-catch' 'dm_control/fish-swim' 'dm_control/finger-spin' 'dm_control/reacher-easy' 'dm_control/reacher-hard' 'dm_control/walker-run'; do
for env in 'dm_control/dog_stand' 'dm_control/humanoid_stand'; do # 'dm_control/dog_run' 'dm_control/acrobot_swingup'; do
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='1' B1='0.9' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=5000 sbatch slurm_experiment_4.sh; # CrossQ
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0007'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=5000 sbatch slurm_experiment_4.sh; # CrossQ
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0007'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=2500 sbatch slurm_experiment_4.sh; # CrossQ
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0004'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=2500 sbatch slurm_experiment_4.sh; # CrossQ
done

