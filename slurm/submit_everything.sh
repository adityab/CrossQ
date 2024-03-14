#!/bin/bash


# # 1 # Main Plots #############
# # Fig 1,3,5
timesteps=5000000
q_bias=0

for env in 'Walker2d-v4' 'Hopper-v4' 'Ant-v4' 'HalfCheetah-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;
    ALGO='td3' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;
done

for env in 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;
    ALGO='td3' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;
done

ALGO='sac' ENV='Hopper-v4' ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=256  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;
ALGO='td3' ENV='Hopper-v4' ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=256  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;


# Baselines
for env in 'Walker2d-v4' 'Hopper-v4' 'Ant-v4' 'HalfCheetah-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1'  PI_DELAY='1'  B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;  # SAC
done

for env in 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1'  PI_DELAY='1'  B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # SAC
done

for env in 'Walker2d-v4' 'Hopper-v4' 'Ant-v4' 'HalfCheetah-v4' 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ALGO='sac'  ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256 N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # SAC(20)
    ALGO='redq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256 N_CRITICS=10 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # REDQ
    ALGO='droq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=1 N_NEURONS=256 N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # DroQ
done


# 2 # Q Bias #############
# # Fig 16
timesteps=1000000
q_bias=1

for env in 'Walker2d-v4' 'Hopper-v4' 'Ant-v4' 'HalfCheetah-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh; # CrossQ
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='1' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh; # SAC
done

for env in 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # CrossQ
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='1' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # SAC
done

for env in 'Walker2d-v4' 'Hopper-v4' 'Ant-v4' 'HalfCheetah-v4' 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ALGO='sac'  ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256 N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # SAC(20)
    ALGO='redq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256 N_CRITICS=10 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # REDQ
    ALGO='droq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=1 N_NEURONS=256 N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # DroQ
done

# 3 # DMC envs #############
# Fig 16
timesteps=1000000
q_bias=0

for env in 'dm_control/ball_in_cup-catch' 'dm_control/fish-swim' 'dm_control/finger-spin' 'dm_control/reacher-easy' 'dm_control/reacher-hard' 'dm_control/walker-run'; do
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh; # CrossQ
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='1' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh; # SAC
    ALGO='redq' ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='1' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256 N_CRITICS=10 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;       # REDQ
done

for env in  'dm_control/humanoid-stand' 'dm_control/humanoid-walk' 'dm_control/humanoid-run'; do
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh; # CrossQ
    ALGO='sac'  ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='1' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256  N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh; # SAC
    ALGO='redq' ENV=$env ACT='relu' LR='0.0008'  UTD='1' PI_DELAY='1' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256  N_CRITICS=10 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh; # REDQ
done

# 4 # CrossQ Ablations #############
# Fig 16
timesteps=1000000
q_bias=0

# TODO
for env in 'Walker2d-v4' 'Hopper-v4' 'Ant-v4' 'HalfCheetah-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=256  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;  # small
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=0 LN=1 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;  # layer norm
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=0 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;  # w/o BN
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.9' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;  # beta_1 = 0.9
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='1' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_multi.sh;  # policy_delay = 1
done

for env in 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=256  N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # small
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=0 LN=1 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # layer norm
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.5' XQ_STYLE=1 BN=0 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # w/o BN
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='3' B1='0.9' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # beta_1 = 0.9
    ALGO='sac' ENV=$env ACT='relu' LR='0.001' UTD='1' PI_DELAY='1' B1='0.5' XQ_STYLE=1 BN=1 LN=0 N_NEURONS=2048 N_CRITICS=2 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment.sh;  # policy_delay = 1
done

# # 5 # REDQ and DroQ Ablations #############
# # Fig 16
timesteps=1000000
q_bias=0

for env in 'Walker2d-v4' 'Hopper-v4' 'Ant-v4' 'HalfCheetah-v4' 'Humanoid-v4' 'HumanoidStandup-v4'; do
    ALGO='redq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256  N_CRITICS=10 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_5.sh;  # REDQ
    ALGO='redq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=2048 N_CRITICS=10 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_5.sh;  # REDQ wide
    ALGO='redq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.5' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=256  N_CRITICS=10 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_5.sh;  # REDQ beta = 0.5
    ALGO='redq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.5' XQ_STYLE=0 BN=0 LN=0 N_NEURONS=2048 N_CRITICS=10 BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_5.sh;  # REDQ beta = 0.5 and wide

    ALGO='droq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=1 N_NEURONS=256  N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_5.sh;  # DroQ
    ALGO='droq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.9' XQ_STYLE=0 BN=0 LN=1 N_NEURONS=2048 N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_5.sh;  # DroQ wide
    ALGO='droq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.5' XQ_STYLE=0 BN=0 LN=1 N_NEURONS=256  N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_5.sh;  # DroQ beta = 0.5
    ALGO='droq' ENV=$env ACT='relu' LR='0.001' UTD='20' PI_DELAY='20' B1='0.5' XQ_STYLE=0 BN=0 LN=1 N_NEURONS=2048 N_CRITICS=2  BN_MODE='brn_actor' BN_MOM='0.99' EVAL_QBIAS=$q_bias STEPS=$timesteps sbatch slurm_experiment_5.sh;  # DroQ beta = 0.5 and wide
done