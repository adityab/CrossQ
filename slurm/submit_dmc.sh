#!/bin/bash

timesteps=500000
q_bias=0

# learning rate 3e-4
# start learning 2500
# policy delay

# for env in 'dm_control/ball_in_cup-catch' 'dm_control/fish-swim' 'dm_control/finger-spin' 'dm_control/reacher-easy' 'dm_control/reacher-hard' 'dm_control/walker-run'; do
# for env in 'dm_control/dog-stand-v0' 'dm_control/humanoid-stand-v0' 'dm_control/pendulum-swingup-v0'; do # 'dm_control/humanoid-stand-v0'; do # 'dm_control/fish-swim' 'dm_control/dog_run' 'dm_control/acrobot_swingup'; do
for env in 'dm_control/acrobot-swingup-v0' 'dm_control/cheetah-run' 'dm_control/dog-run-v0' 'dm_control/dog-stand-v0' 'dm_control/dog-trot-v0' 'dm_control/dog-walk-v0' 'dm_control/finger-turn-hard-v0' 'dm_control/hopper-hop-v0' \
'dm_control/humanoid-stand-v0' 'dm_control/humanoid-walk-v0' 'dm_control/quadruped-run-v0' 'dm_control/walker-walk-v0'; do
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=2500 sbatch slurm_experiment_multi.sh; # CrossQ
    # ALGO='sac'  ENV=$env ACT='relu' LR='0.0007'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=5000 sbatch slurm_experiment_4.sh; # CrossQ
    # ALGO='sac'  ENV=$env ACT='relu' LR='0.0007'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=2500 sbatch slurm_experiment_4.sh; # CrossQ
    # ALGO='sac'  ENV=$env ACT='relu' LR='0.0004'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=2500 sbatch slurm_experiment_4.sh; # CrossQ
done

for env in 'dm_control/acrobot-swingup-v0' 'dm_control/dog-run-v0' 'dm_control/finger-turn-hard-v0' 'dm_control/hopper-hop-v0' 'dm_control/humanoid-stand-v0'; do
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.9' EVAL_QBIAS=$q_bias STEPS=$timesteps START=2500 sbatch slurm_experiment_multi.sh; # CrossQ
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.999' EVAL_QBIAS=$q_bias STEPS=$timesteps START=2500 sbatch slurm_experiment_multi.sh; # CrossQ
    # ALGO='sac'  ENV=$env ACT='relu' LR='0.0007'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=5000 sbatch slurm_experiment_4.sh; # CrossQ
    # ALGO='sac'  ENV=$env ACT='relu' LR='0.0007'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=2500 sbatch slurm_experiment_4.sh; # CrossQ
    # ALGO='sac'  ENV=$env ACT='relu' LR='0.0004'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps START=2500 sbatch slurm_experiment_4.sh; # CrossQ
done

