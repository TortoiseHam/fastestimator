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
import math
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as fn

import fastestimator as fe
from fastestimator.dataset.data import cifair100
from fastestimator.op.numpyop.univariate import ChannelTranspose, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace
from fastestimator.trace.io import BestModelSaver, Traceability
from fastestimator.trace.metric import MCC
from fastestimator.util import Data


def get_gaussian_filter(kernel_size=3, sigma=2.0, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3:
        padding = 1
    else:
        padding = 0
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class CBSLeNet(torch.nn.Module):
    def __init__(self, input_shape=(3, 32, 32), classes=100) -> None:
        super().__init__()
        self.pool_kernel = 2
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        flat_x = input_shape[1] // self.pool_kernel ** 2
        flat_y = input_shape[2] // self.pool_kernel ** 2
        self.fc1 = nn.Linear(flat_x * flat_y * 64, 64)
        self.fc2 = nn.Linear(64, classes)
        self.std = 1 / 0.9
        self.get_new_kernels()

    def get_new_kernels(self):
        self.std *= 0.9
        self.kernel1 = get_gaussian_filter(kernel_size=3, sigma=self.std, channels=32)
        self.kernel2 = get_gaussian_filter(kernel_size=3, sigma=self.std, channels=64)
        self.kernel3 = get_gaussian_filter(kernel_size=3, sigma=self.std, channels=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.kernel1(x)
        x = fn.relu(fn.max_pool2d(x, self.pool_kernel))

        x = self.conv2(x)
        x = self.kernel2(x)
        x = fn.relu(fn.max_pool2d(x, self.pool_kernel))

        x = self.conv3(x)
        x = self.kernel3(x)
        x = fn.relu(x)

        x = x.view(x.size(0), -1)
        x = fn.relu(self.fc1(x))
        x = fn.softmax(self.fc2(x), dim=-1)
        return x


class Adjuster(Trace):
    def __init__(self, model):
        super().__init__(mode='train')
        self.model = model

    def on_epoch_begin(self, data: Data) -> None:
        if self.system.epoch_idx % 5 == 0:
            self.model.get_new_kernels(self.system.epoch_idx)


def get_estimator(epochs=50,
                  batch_size=64,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    # step 1
    train_data, eval_data = cifair100.load_data()
    test_data = eval_data.split(0.5, stratify="y", seed=42)
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            ChannelTranspose(inputs="x", outputs="x"),
        ],
        num_process=0)

    # step 2
    model = fe.build(model_fn=lambda: CBSLeNet(), optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce"),
    ])

    # step 3
    traces = [
        MCC(true_key="y", pred_key="y_pred", output_name="mcc"),
        BestModelSaver(model=model, save_dir=save_dir, metric="mcc", save_best_mode="max", load_best_final=True),
        Adjuster(model),
        Traceability('CBS')
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit('cbs')
    est.test()
