from functools import partial
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState, ActorTrainState
from sbx.sac.policies import SACPolicy


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> float:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.ent_coef_init))
        return self.ent_coef_init


class SAC(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        # Minimal dict support using flatten()
        "MultiInputPolicy": SACPolicy,
    }

    policy: SACPolicy
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        qf_learning_rate: Optional[float] = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        crossq_style: bool = False,
        td3_mode: bool = False,
        use_bnstats_from_live_net: bool = False,
        policy_q_reduce_fn = jnp.min,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
        stats_window_size: int = 100,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            stats_window_size=stats_window_size,
        )

        self.policy_delay = policy_delay
        self.ent_coef_init = ent_coef
        self.crossq_style = crossq_style
        self.td3_mode = td3_mode
        self.use_bnstats_from_live_net = use_bnstats_from_live_net
        self.policy_q_reduce_fn = policy_q_reduce_fn

        if td3_mode:
            self.action_noise = NormalActionNoise(mean=jnp.zeros(1), sigma=jnp.ones(1) * 0.1)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            # pytype: disable=not-instantiable
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                td3_mode=self.td3_mode,
                **self.policy_kwargs,
            )
            # pytype: enable=not-instantiable

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.actor = self.policy.actor  # type: ignore[assignment]
            self.qf = self.policy.qf  # type: ignore[assignment]

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith("auto"):
                # Default initial value of ent_coef when learned
                ent_coef_init = 1.0
                if "_" in self.ent_coef_init:
                    ent_coef_init = float(self.ent_coef_init.split("_")[1])
                    assert ent_coef_init > 0.0, "The initial value of ent_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.ent_coef = EntropyCoef(ent_coef_init)
            else:
                # This will throw an error if a malformed string (different from 'auto') is passed
                assert isinstance(
                    self.ent_coef_init, float
                ), f"Entropy coef must be float when not equal to 'auto', actual: {self.ent_coef_init}"
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_init)  # type: ignore[assignment]

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init(ent_key)["params"],
                tx=optax.adam(
                    learning_rate=self.learning_rate,
                ),
            )

        # automatically set target entropy if needed
        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, batch_size, gradient_steps):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        # Pre-compute the indices where we need to update the actor
        # This is a hack in order to jit the train loop
        # It will compile once per value of policy_delay_indices
        policy_delay_indices = {i: True for i in range(gradient_steps) if ((self._n_updates + i + 1) % self.policy_delay) == 0}
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            log_metrics,
        ) = self._train(
            self.crossq_style,
            self.td3_mode,
            self.use_bnstats_from_live_net,
            self.gamma,
            self.tau,
            self.target_entropy,
            gradient_steps,
            data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            self.policy_q_reduce_fn,
        )
        self._n_updates += gradient_steps
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for k,v in log_metrics.items():
            self.logger.record(f"train/{k}", v.item())
    
    @staticmethod
    @partial(jax.jit, static_argnames=["crossq_style", "td3_mode", "use_bnstats_from_live_net"])
    def update_critic(
        crossq_style: bool,
        td3_mode: bool,
        use_bnstats_from_live_net: bool,
        gamma: float,
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        key: jax.random.KeyArray,
    ):
        key, noise_key, dropout_key_target, dropout_key_current, redq_key = jax.random.split(key, 5)
        # sample action from the actor
        dist = actor_state.apply_fn(
            {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
            next_observations, train=False
        )

        if td3_mode:
            ent_coef_value = 0.0
            target_policy_noise = 0.2
            target_noise_clip = 0.5

            next_state_actions = dist.mode()
            noise = jax.random.normal(noise_key, next_state_actions.shape) * target_policy_noise
            noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
            next_state_actions = jnp.clip(next_state_actions + noise, -1.0, 1.0)
            next_log_prob = jnp.zeros(next_state_actions.shape[0])
        else:
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

            next_state_actions = dist.sample(seed=noise_key)
            next_log_prob = dist.log_prob(next_state_actions)

        def mse_loss(params, batch_stats, dropout_key):
            if not crossq_style:
                next_q_values = qf_state.apply_fn(
                    {
                        "params": qf_state.target_params, 
                        "batch_stats": qf_state.target_batch_stats if not use_bnstats_from_live_net else batch_stats
                    },
                    next_observations, next_state_actions,
                    rngs={"dropout": dropout_key_target},
                    train=False
                )

                # shape is (n_critics, batch_size, 1)
                current_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats}, 
                    observations, actions, 
                    rngs={"dropout": dropout_key}, 
                    mutable=["batch_stats"],
                    train=True,
                )

            else:
                # ----- CrossQ's One Weird Trick™ -----
                # concatenate current and next observations to double the batch size
                # new shape of input is (n_critics, 2*batch_size, obs_dim + act_dim)
                # apply critic to this bigger batch
                catted_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats}, 
                    jnp.concatenate([observations, next_observations], axis=0), 
                    jnp.concatenate([actions, next_state_actions], axis=0), 
                    rngs={"dropout": dropout_key}, 
                    mutable=["batch_stats"],
                    train=True,
                )
                current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)
            
            if next_q_values.shape[0] > 2: # only for REDQ
                # REDQ style subsampling of critics.
                m_critics = 2  
                next_q_values = jax.random.choice(redq_key, next_q_values, (m_critics,), replace=False, axis=0)

            next_q_values = jnp.min(next_q_values, axis=0)
            next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(-1, 1)
            target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_q_values  # shape is (batch_size, 1)

            loss = 0.5 * ((jax.lax.stop_gradient(target_q_values) - current_q_values) ** 2).mean(axis=1).sum()

            return loss, (state_updates, current_q_values, next_q_values)
        
        (qf_loss_value, (state_updates, current_q_values, next_q_values)), grads = \
            jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, qf_state.batch_stats, dropout_key_current)
        
        qf_state = qf_state.apply_gradients(grads=grads)
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])

        metrics = {
            'critic_loss': qf_loss_value, 
            'ent_coef': ent_coef_value, 
            'current_q_values': current_q_values.mean(), 
            'next_q_values': next_q_values.mean(),
        }

        return (qf_state, metrics, key)

    @staticmethod
    @partial(jax.jit, static_argnames=["q_reduce_fn", "td3_mode"])
    def update_actor(
        actor_state: ActorTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: np.ndarray,
        key: jax.random.KeyArray,
        q_reduce_fn = jnp.min,  # Changes for redq and droq
        td3_mode = False,
    ):
        key, dropout_key, noise_key, redq_key = jax.random.split(key, 4)

        def actor_loss(params, batch_stats):
            dist, state_updates = actor_state.apply_fn({
                "params": params, "batch_stats": batch_stats},
                observations,
                mutable=["batch_stats"], 
                train=True
            )

            if td3_mode:
                actor_actions = dist.mode()
                ent_coef_value = 0.0
                log_prob = jnp.zeros(actor_actions.shape[0])
            else:
                actor_actions = dist.sample(seed=noise_key)
                ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
                log_prob = dist.log_prob(actor_actions).reshape(-1, 1)

            qf_pi = qf_state.apply_fn(
                {
                    "params": qf_state.params,
                    "batch_stats": qf_state.batch_stats
                },
                observations,
                actor_actions,
                rngs={"dropout": dropout_key}, train=False
            )
            
            min_qf_pi = q_reduce_fn(qf_pi, axis=0)

            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, (-log_prob.mean(), state_updates)

        (actor_loss_value, (entropy, state_updates)), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params, actor_state.batch_stats)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(batch_stats=state_updates["batch_stats"])

        return actor_state, qf_state, actor_loss_value, key, entropy

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf_state: RLTrainState):
        qf_state = qf_state.replace(target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        qf_state = qf_state.replace(target_batch_stats=optax.incremental_update(qf_state.batch_stats, qf_state.target_batch_stats, tau))
        return qf_state

    @staticmethod
    @jax.jit
    def update_temperature(target_entropy: np.ndarray, ent_coef_state: TrainState, entropy: float):
        def temperature_loss(temp_params):
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
            ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    @classmethod
    @partial(jax.jit, static_argnames=["cls", "crossq_style", "td3_mode", "use_bnstats_from_live_net", "gradient_steps", "q_reduce_fn"])
    def _train(
        cls,
        crossq_style: bool,
        td3_mode: bool,
        use_bnstats_from_live_net: bool,
        gamma: float,
        tau: float,
        target_entropy: np.ndarray,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        policy_delay_indices: flax.core.FrozenDict,
        qf_state: RLTrainState,
        actor_state: ActorTrainState,
        ent_coef_state: TrainState,
        key,
        q_reduce_fn,
    ):
        actor_loss_value = jnp.array(0)

        for i in range(gradient_steps):

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step : batch_size * (step + 1)]

            (
                qf_state,
                log_metrics_critic,
                key,
            ) = SAC.update_critic(
                crossq_style,
                td3_mode,
                use_bnstats_from_live_net,
                gamma,
                actor_state,
                qf_state,
                ent_coef_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                key,
            )
            qf_state = SAC.soft_update(tau, qf_state)

            # hack to be able to jit (n_updates % policy_delay == 0)
            if i in policy_delay_indices:
                (actor_state, qf_state, actor_loss_value, key, entropy) = cls.update_actor(
                    actor_state,
                    qf_state,
                    ent_coef_state,
                    slice(data.observations),
                    key,
                    q_reduce_fn,
                    td3_mode,
                )
                ent_coef_state, _ = SAC.update_temperature(target_entropy, ent_coef_state, entropy)

        log_metrics = {
            'actor_loss' : actor_loss_value,
            **log_metrics_critic
        }

        return (
            qf_state,
            actor_state,
            ent_coef_state,
            key,
            log_metrics,
        )
    
    def predict_critic(self, observation, action):
        return self.policy.predict_critic(observation, action)
    
    def current_entropy_coeff(self):
        return self.ent_coef_state.apply_fn({"params": self.ent_coef_state.params})