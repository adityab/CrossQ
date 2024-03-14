import argparse
import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
import jax
import jax.numpy as jnp
import rlax
import flax.linen as nn

from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization
from sbx import SAC
from sbx.sac.actor_critic_evaluation_callback import CriticBiasCallback, EvalCallback
from sbx.sac.utils import *

import gymnasium as gym
from shimmy.registration import DM_CONTROL_SUITE_ENVS


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['WANDB_DIR'] = '/tmp'

parser = argparse.ArgumentParser()
parser.add_argument("-env",         type=str, required=False, default="HumanoidStandup-v4", help="Set Environment.")
parser.add_argument("-algo",        type=str, required=False, default='sac', choices=['crossq', 'sac', 'redq', 'droq', 'td3'], help="critic activation function")
parser.add_argument("-seed",        type=int, required=False, default=1, help="Set Seed.")
parser.add_argument("-log_freq",    type=int, required=False, default=300, help="how many times to log during training")

parser.add_argument('-wandb_project', type=str, required=False, default='crossQ', help='wandb project name')
parser.add_argument("-wandb_mode",    type=str, required=False, default='disabled', choices=['disabled', 'online'], help="enable/disable wandb logging")
parser.add_argument("-eval_qbias",    type=int, required=False, default=0, choices=[0,1], help="enable/diasble q bias evaluation (expensive)")

parser.add_argument("-adam_b1",           type=float, required=False, default=0.5, help="adam b1 parameter")
parser.add_argument("-bn",                type=float, required=False, default=False,  choices=[0,1], help="Use batch norm layers in the actor and critic networks")
parser.add_argument("-bn_momentum",       type=float, required=False, default=0.99, help="batch norm momentum parameter")
parser.add_argument("-bn_mode",           type=str,   required=False, default='brn_actor', help="batch norm mode (bn or brn)")
parser.add_argument("-critic_activation", type=str,   required=False, default='relu', help="critic activation function")
parser.add_argument("-crossq_style",      type=float, required=False, default=1,choices=[0,1], help="crossq style joint forward pass through critic network")
parser.add_argument("-dropout",           type=int,   required=False, default=0, choices=[0,1], help="whether to use dropout for SAC")
parser.add_argument("-ln",                type=float, required=False, default=False, choices=[0,1], help="layernorm in critic network")
parser.add_argument("-lr",                type=float, required=False, default=1e-3, help="actor and critic learning rate")
parser.add_argument("-n_critics",         type=int,   required=False, default=2, help="number of critics to use")
parser.add_argument("-n_neurons",         type=int,   required=False, default=256, help="number of neurons for each critic layer)")
parser.add_argument("-policy_delay",      type=int,   required=False, default=1, help="policy is updated after this many critic updates")
parser.add_argument("-tau",               type=float, required=False, default=0.005, help="target network averaging")
parser.add_argument("-utd",               type=int,   required=False, default=1, help="number of critic updates per env step")
parser.add_argument("-total_timesteps",   type=int,   required=False, default=5e6, help="total number of training steps")

parser.add_argument("-bnstats_live_net",  type=int,   required=False, default=0,choices=[0,1], help="use bn running statistics from live network within the target network")

experiment_time = time.time()
args = parser.parse_args()

seed = args.seed
args.algo = str.lower(args.algo)
args.bn = bool(args.bn)
args.crossq_style = bool(args.crossq_style)
args.tau = float(args.tau) if not args.crossq_style else 1.0
args.bn_momentum = float(args.bn_momentum) if args.bn else 0.0
dropout_rate, layer_norm = None, False
policy_q_reduce_fn = jax.numpy.min
net_arch = {'pi': [256, 256], 'qf': [args.n_neurons, args.n_neurons]}

total_timesteps = int(args.total_timesteps)
eval_freq = max(5_000_000 // args.log_freq, 1)

if 'dm_control' in args.env:
    total_timesteps = {
        'dm_control/reacher-easy'     : 100_000,
        'dm_control/reacher-hard'     : 100_000,
        'dm_control/ball_in_cup-catch': 200_000,
        'dm_control/finger-spin'      : 500_000,
        'dm_control/fish-swim'        : 5_000_000,
        'dm_control/humanoid-stand'   : 5_000_000,
    }[args.env]
    eval_freq = max(total_timesteps // args.log_freq, 1)

td3_mode = False

if args.algo == 'droq':
    dropout_rate = 0.01
    layer_norm = True
    policy_q_reduce_fn = jax.numpy.mean
    args.n_critics = 2
    # args.adam_b1 = 0.9  # adam default
    args.adam_b2 = 0.999  # adam default
    args.policy_delay = 20
    args.utd = 20
    group = f'DroQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

elif args.algo == 'redq':
    policy_q_reduce_fn = jax.numpy.mean
    args.n_critics = 10
    # args.adam_b1 = 0.9  # adam default
    args.adam_b2 = 0.999  # adam default
    args.policy_delay = 20
    args.utd = 20
    group = f'REDQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

elif args.algo == 'td3':
    # With the right hyperparameters, this here can run all the above algorithms
    # and ablations.
    td3_mode = True
    layer_norm = args.ln
    if args.dropout: 
        dropout_rate = 0.01
    group = f'TD3_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

elif args.algo == 'sac':
    # With the right hyperparameters, this here can run all the above algorithms
    # and ablations.
    layer_norm = args.ln
    if args.dropout: 
        dropout_rate = 0.01
    group = f'SAC_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

elif args.algo == 'crossq':
    args.adam_b1 = 0.5
    args.policy_delay = 3
    args.n_critics = 2
    args.utd = 1                    # nice
    net_arch["qf"] = [2048, 2048]   # wider critics
    args.bn = True                  # use batch norm
    args.crossq_style = True        # with a joint forward pass
    args.tau = 1.0                  # without target networks
    group = f'CrossQ_{args.env}'

else:
    raise NotImplemented

args_dict = vars(args)
args_dict.update({
    "dropout_rate": dropout_rate,
    "layer_norm": layer_norm
})

with wandb.init(
    entity='your_entity',
    project=args.wandb_project,
    name=f"seed={seed}",
    group=group,
    tags=[],
    sync_tensorboard=True,
    config=args_dict,
    settings=wandb.Settings(start_method="fork") if is_slurm_job() else None,
    mode=args.wandb_mode
) as wandb_run:
    
    # SLURM maintainance
    if is_slurm_job():
        print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
        wandb_run.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')

    training_env = gym.make(args.env)

    if args.env == 'dm_control/humanoid-stand':
        training_env.observation_space['head_height'] = gym.spaces.Box(-np.inf, np.inf, (1,))
    if args.env == 'dm_control/fish-swim':
        training_env.observation_space['upright'] = gym.spaces.Box(-np.inf, np.inf, (1,))

    import optax
    model = SAC(
        "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        training_env,
        policy_kwargs=dict({
            'activation_fn': activation_fn[args.critic_activation],
            'layer_norm': layer_norm,
            'batch_norm': bool(args.bn),
            'batch_norm_momentum': float(args.bn_momentum),
            'batch_norm_mode': args.bn_mode,
            'dropout_rate': dropout_rate,
            'n_critics': args.n_critics,
            'net_arch': net_arch,
            'optimizer_class': optax.adam,
            'optimizer_kwargs': dict({
                'b1': args.adam_b1,
                'b2': 0.999 # default
            })
        }),
        gradient_steps=args.utd,
        policy_delay=args.policy_delay,
        crossq_style=bool(args.crossq_style),
        td3_mode=td3_mode,
        use_bnstats_from_live_net=bool(args.bnstats_live_net),
        policy_q_reduce_fn=policy_q_reduce_fn,
        learning_starts=5000,
        learning_rate=args.lr,
        qf_learning_rate=args.lr,
        tau=args.tau,
        gamma=0.99 if not args.env == 'Swimmer-v4' else 0.9999,
        verbose=0,
        buffer_size=1_000_000,
        seed=seed,
        stats_window_size=1,  # don't smooth the episode return stats over time
        tensorboard_log=f"logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/",
    )

    # Create log dir where evaluation results will be saved
    eval_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/eval/"
    qbias_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/qbias/"
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(qbias_log_dir, exist_ok=True)

    # Create callback that evaluates agent
    eval_callback = EvalCallback(
        make_vec_env(args.env, n_envs=1, seed=seed),
        jax_random_key_for_seeds=args.seed,
        best_model_save_path=None,
        log_path=eval_log_dir, eval_freq=eval_freq,
        n_eval_episodes=1, deterministic=True, render=False
    )

    # Callback that evaluates q bias according to the REDQ paper.
    q_bias_callback = CriticBiasCallback(
        make_vec_env(args.env, n_envs=1, seed=seed), 
        jax_random_key_for_seeds=args.seed,
        best_model_save_path=None,
        log_path=qbias_log_dir, eval_freq=eval_freq,
        n_eval_episodes=1, render=False
    )

    callback_list = CallbackList(
        [eval_callback, q_bias_callback, WandbCallback(verbose=0,)] if args.eval_qbias else 
        [eval_callback, WandbCallback(verbose=0,)]
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback_list)
