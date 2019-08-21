# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
from tensorflow.python.framework import ops as tfops

from fastestimator.network.model import ModelOp
from fastestimator.util.op import get_op_from_mode, verify_ops, get_inputs_by_op, write_outputs_by_key
from fastestimator.util.schedule import Scheduler
from fastestimator.util.util import NonContext


class Network:
    def __init__(self, ops):
        if not isinstance(ops, list):
            ops = [ops]
        self.ops = ops
        self.model_schedule = {}
        self.op_schedule = {}
        self.current_epoch_ops = {}
        self.current_epoch_model = {}

    def prepare(self, mode_list):
        for mode in mode_list:
            signature_epoch, mode_ops = self._get_signature_epoch(mode)
            epoch_ops_map = {}
            epoch_model_map = {}
            for epoch in signature_epoch:
                epoch_ops = []
                epoch_model = []
                # generate ops for specific mode and epoch
                for op in mode_ops:
                    if isinstance(op, Scheduler):
                        scheduled_op = op.get_current_value(epoch)
                        if scheduled_op:
                            epoch_ops.append(scheduled_op)
                    else:
                        epoch_ops.append(op)
                # check the ops
                verify_ops(epoch_ops, "Network")
                # create model list
                for op in epoch_ops:
                    if isinstance(op, ModelOp) and op.model not in epoch_model:
                        epoch_model.append(op.model)
                assert epoch_model, "Network has no model for epoch {}".format(epoch)
                epoch_ops_map[epoch] = epoch_ops
                epoch_model_map[epoch] = epoch_model
            self.op_schedule[mode] = Scheduler(epoch_dict=epoch_ops_map)
            self.model_schedule[mode] = Scheduler(epoch_dict=epoch_model_map)

    def _get_signature_epoch(self, mode):
        signature_epoch = [0]
        mode_ops = get_op_from_mode(self.ops, mode)
        for op in mode_ops:
            if isinstance(op, Scheduler):
                signature_epoch.extend(op.keys)
        return list(set(signature_epoch)), mode_ops

    def load_epoch(self, epoch, mode):
        ops = self.op_schedule[mode].get_current_value(epoch)
        model_list = self.model_schedule[mode].get_current_value(epoch)
        return ops, model_list

    def run_step(self, batch, ops, model_list, state, warm_up=False):
        mode = state["mode"]
        num_model = len(model_list)
        # use gradient tape for train, otherwise use a dummy tape
        with tf.GradientTape(persistent=True) if mode == "train" else NonContext() as tape:
            state['tape'] = tape
            Network._forward(batch, state, ops)
            losses = Network._forward(batch, state, map(lambda x: x.loss, model_list), recycle=False, ret_response=True)
        # update model only for train mode
        if mode == "train":
            for idx in range(num_model):
                gradients = tape.gradient(losses[idx], model_list[idx].trainable_variables)
                if warm_up:
                    with tfops.init_scope():
                        _ = model_list[idx].optimizer.iterations
                        model_list[idx].optimizer._create_hypers()
                        model_list[idx].optimizer._create_slots(model_list[idx].trainable_variables)
                else:
                    model_list[idx].optimizer.apply_gradients(zip(gradients, model_list[idx].trainable_variables))
        del state['tape']
        del tape
        return losses

    @staticmethod
    def _forward(batch, state, ops, recycle=True, ret_response=False):
        response = ()
        data = None
        for op in ops:
            if not recycle:
                data = None
            data = get_inputs_by_op(op, batch, data)
            data = op.forward(data, state)
            if op.outputs:
                write_outputs_by_key(batch, data, op.outputs)
            if ret_response:
                response += data,
        return response
