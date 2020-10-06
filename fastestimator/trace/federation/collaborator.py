#  Copyright 2020 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
from typing import List, Optional, Union, TypeVar

import tensorflow as tf
import torch

from fastestimator.trace.trace import Trace
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list, parse_freq
from fastestimator.util.data import Data

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)
Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)


@traceable()
class Collaborator(Trace):
    """A class to make this node a collaborator within a Federated Learning cluster.

    Args:
        frequency: 'batch', 'epoch', integer, or strings like '10s', '15e'. When using 'batch', the training pauses to
            communicate with the aggregator after every batch. The same applies for 'epoch'. If using an integer, let's
            say 1000, the training will pause after every 1000 steps. You can also use strings like '8s' to indicate
            every 8 steps or '5e' to indicate every 5 epochs. Higher frequencies will dramatically slow down training.
        models: Which models are participating in the federation. Their weights will be sent to and subsequently updated
            by the aggregator.
    """
    def __init__(self, frequency: Union[int, str] = '1e', models: Union[None, Model, List[Model]] = None):
        super().__init__(mode='train')
        self.models = to_list(models)
        self.frequency = parse_freq(frequency)

    def on_batch_end(self, data: Data) -> None:
        if self.frequency.freq and self.frequency.is_step and self.system.global_step % self.frequency.freq == 0:
            self._send_models()
            self._update_models()

    def on_epoch_end(self, data: Data) -> None:
        if self.frequency.freq and not self.frequency.is_step and self.system.epoch_idx % self.frequency.freq == 0:
            self._send_models()
            self._update_models()

    def _send_models(self) -> None:
        """Send the locally updated model weights to the Aggregator.
        """
        pass

    def _update_models(self) -> None:
        """Get new model weights from the Aggregator and apply them.
        """
        pass
