from typing import NamedTuple, Any

import flax
import numpy as np
from flax.training.train_state import TrainState

class ActorTrainState(TrainState):
    batch_stats: flax.core.FrozenDict

class RLTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]
    batch_stats: flax.core.FrozenDict
    target_batch_stats: flax.core.FrozenDict


class ReplayBufferSamplesNp(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
