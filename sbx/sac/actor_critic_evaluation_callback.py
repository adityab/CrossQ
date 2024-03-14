import warnings
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization
from stable_baselines3.common import type_aliases
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, EventCallback, BaseCallback
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import gymnasium as gym
import jax
import os
import rlax


def get_mc_return_with_entropy_on_reset(bias_eval_env, model, max_ep_len, gamma, n_mc_eval, n_mc_cutoff):
    # Based off of original REDQ analysis function and adapted for JAX.
    # https://github.com/watchernyu/REDQ/blob/45a28d3e0960e222d1864c2cc8dabc1af287839f/redq/utils/bias_utils.py#L6

    # since we want to also compute bias, so we need to
    final_mc_list = np.zeros(0)
    final_mc_entropy_list = np.zeros(0)
    final_obs_list = []
    final_act_list = []
    while final_mc_list.shape[0] < n_mc_eval:
        # we continue if haven't collected enough data
        o = bias_eval_env.reset()
        # temporary lists
        reward_list, log_prob_a_tilda_list, obs_list, act_list = [], [], [], []
        r, d, ep_ret, ep_len = 0, False, 0, 0
        for _ in range(max_ep_len):  # run an episode
            a, log_prob_a_tilda = model.policy.predict_action_with_logprobs(
                o,  # type: ignore[arg-type]
                deterministic=False,
            )

            obs_list.append(o)
            act_list.append(a)
            o, r, d, _ = bias_eval_env.step(a)
            ep_len += 1
            reward_list.append(r)
            log_prob_a_tilda_list.append(log_prob_a_tilda)
            if d or (ep_len == max_ep_len):
                break

        rewards = jnp.concatenate(reward_list)
        log_probs = jnp.concatenate(log_prob_a_tilda_list)
        alpha = model.current_entropy_coeff()

        @jax.jit
        def calc_disc_returns(rewards, alpha, log_probs):
            soft_rewards = rewards - alpha * jnp.concatenate([log_probs[:-1], jnp.zeros(1)])
            gammas = jnp.ones_like(rewards) * gamma
            discounted_return_list = rlax.discounted_returns(rewards, gammas, 0.0)
            discounted_return_with_entropy_list = rlax.discounted_returns(soft_rewards, gammas, 0.0)
            return discounted_return_list, discounted_return_with_entropy_list

        discounted_return_list, discounted_return_with_entropy_list = calc_disc_returns(
            rewards, alpha, log_probs
        )

        # now we take the first few of these.
        final_mc_list = jnp.concatenate((final_mc_list, discounted_return_list[:n_mc_cutoff]))
        final_mc_entropy_list = jnp.concatenate((final_mc_entropy_list, discounted_return_with_entropy_list[:n_mc_cutoff]))
        final_obs_list += obs_list[:n_mc_cutoff]
        final_act_list += act_list[:n_mc_cutoff]

    final_obs_list = jnp.concatenate(final_obs_list)
    final_act_list = jnp.concatenate(final_act_list)

    q_prediction = jnp.mean(model.predict_critic(
        final_obs_list,
        final_act_list
    ), axis=0)[..., 0]

    @jax.jit
    def calc_metrics(q_prediction, final_mc_entropy_list):
        bias = q_prediction - final_mc_entropy_list
        bias_abs = jnp.abs(bias)
        bias_squared = bias ** 2
        final_mc_entropy_list_normalize_base = jnp.abs(final_mc_entropy_list)
        final_mc_entropy_list_normalize_base = jnp.clip(final_mc_entropy_list_normalize_base, a_min=10.)
        normalized_bias_per_state = bias / final_mc_entropy_list_normalize_base
        normalized_bias_sqr_per_state = bias_squared / final_mc_entropy_list_normalize_base
        return dict(
            MCDisRet=final_mc_list,
            MCDisRetEnt=final_mc_entropy_list,
            QPred=q_prediction,
            QBias=bias,
            QBiasAbs=bias_abs,
            QBiasSqr=bias_squared,
            NormQBias=normalized_bias_per_state,
            NormQBiasSqr=normalized_bias_sqr_per_state
        )

    return calc_metrics(q_prediction, final_mc_entropy_list)


class CriticBiasCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        jax_random_key_for_seeds: int,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )

        self.final_mc_list = []
        self.final_mc_entropy_list = []
        self.q_prediction = []
        self.bias = []
        self.bias_abs = []
        self.bias_squared = []
        self.normalized_bias_per_state = []
        self.normalized_bias_sqr_per_stat = []

        # generate a list of 1M random integers, using a jax random key supplied in the args
        seed_list = jax.random.randint(jax.random.PRNGKey(jax_random_key_for_seeds), (10000000,), 0, 2 ** 30 - 1)
        # cast to numpy
        self.seed_list = np.array(seed_list)
        

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0 or self.n_calls == 1):
            # reset the env with a new seed at the current timestep (reproducibilty)
            self.eval_env.seed(int(self.seed_list[self.n_calls]))
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            key, noise_key = self.model.policy.key, self.model.policy.noise_key            
            metrics = get_mc_return_with_entropy_on_reset(
                self.eval_env, self.model, 1000, 0.99, n_mc_eval=1000, n_mc_cutoff=350
            )
            # Reset random key, to before evaluation. 
            # This way we make sure that the randomness in the main training is not
            # influenced by the evaluation, how it should be.
            self.model.policy.key, self.model.policy.noise_key = key, noise_key


            if self.log_path is not None:

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    **metrics,
                    **kwargs,
                )

            for k,v in metrics.items():
                self.logger.record(f"Q/{k}_mean", jnp.mean(v).item())
                self.logger.record(f"Q/{k}_std", jnp.std(v).item())

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training



class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        jax_random_key_for_seeds: int,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

        # generate a list of 1M random integers, using a jax random key supplied in the args
        seed_list = jax.random.randint(jax.random.PRNGKey(jax_random_key_for_seeds), (10000000,), 0, 2 ** 30 - 1)
        # cast to numpy
        self.seed_list = np.array(seed_list)


    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0 or self.n_calls == 1):
            # reset the env with a new seed at the current timestep (reproducibilty)
            self.eval_env.seed(int(self.seed_list[self.n_calls]))

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)