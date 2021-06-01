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
import tempfile
from typing import List, Tuple, Dict, Any, Union

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import layers
from tensorflow_addons.image import gaussian_filter2d

import fastestimator as fe
from fastestimator.dataset.data import cifair100
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize, RUA
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler
from fastestimator.trace.io import BestModelSaver, Traceability
from fastestimator.trace.metric import MCC, Accuracy
from fastestimator.trace import Trace
from fastestimator.util import Data
from fastestimator.trace.adapt import LRScheduler, EarlyStopping

import inspect
from typing import Callable, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.get_lr import get_lr
from fastestimator.backend.set_lr import set_lr
from fastestimator.schedule.lr_shedule import ARC
from fastestimator.summary.system import System
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable


@traceable()
class DecayScheduler(Trace):
    """Learning rate scheduler trace that changes the learning rate while training.

    This class requires an input function which takes either 'epoch' or 'step' as input:
    ```python
    s = LRScheduler(model=model, lr_fn=lambda step: fe.schedule.cosine_decay(step, cycle_length=3750, init_lr=1e-3))
    fe.Estimator(..., traces=[s])  # Learning rate will change based on step
    s = LRScheduler(model=model, lr_fn=lambda epoch: fe.schedule.cosine_decay(epoch, cycle_length=3750, init_lr=1e-3))
    fe.Estimator(..., traces=[s])  # Learning rate will change based on epoch
    ```

    Args:
        model: A model instance compiled with fe.build.
        lr_fn: A lr scheduling function that takes either 'epoch' or 'step' as input, or the string 'arc'.

    Raises:
        AssertionError: If the `lr_fn` is not configured properly.
    """
    system: System

    def __init__(self, model: Union[tf.keras.Model, torch.nn.Module], lr_fn: Callable[[int], float]) -> None:
        self.model = model
        self.lr_fn = lr_fn
        arg = list(inspect.signature(lr_fn).parameters.keys())
        assert len(arg) == 1 and arg[0] in {"step", "epoch"}, "the lr_fn input arg must be either 'step' or 'epoch'"
        self.schedule_mode = arg[0]
        super().__init__(outputs=self.model.model_name + "_weight_decay")

    def on_epoch_begin(self, data: Data) -> None:
        if self.system.mode == "train" and self.schedule_mode == "epoch":
            new_lr = np.float32(self.lr_fn(self.system.epoch_idx))
            tf.keras.backend.set_value(self.model.current_optimizer.weight_decay, new_lr)

    def on_batch_begin(self, data: Data) -> None:
        if self.system.mode == "train" and self.schedule_mode == "step":
            new_lr = np.float32(self.lr_fn(self.system.global_step))
            tf.keras.backend.set_value(self.model.current_optimizer.weight_decay, new_lr)

    def on_batch_end(self, data: Data) -> None:
        if self.system.mode == "train" and self.system.log_steps and (
                self.system.global_step % self.system.log_steps == 0 or self.system.global_step == 1):
            current_lr = np.float32(tf.keras.backend.get_value(self.model.current_optimizer.weight_decay))
            data.write_with_log(self.outputs[0], current_lr)


class GaussianBlur(layers.Layer):
    def __init__(self, sigma: tf.Variable):
        super().__init__()
        self.sigma = sigma

    def get_config(self) -> Dict[str, Any]:
        return {'sigma': self.sigma}

    def call(self, x: Union[tf.Tensor, List[tf.Tensor]], **kwargs) -> tf.Tensor:
        # return gaussian_filter2d(x, sigma=self.sigma, padding='CONSTANT')
        return gaussian_blur(x, sigma=self.sigma)


def CBSLeNet2(sigma: tf.Variable, input_size: Tuple[int, int, int] = (32, 32, 3), classes: int = 100) -> tf.keras.Model:
    """A small 9-layer ResNet Tensorflow model for cifar10 image classification.
    The model architecture is from https://github.com/davidcpage/cifar10-fast

    Args:
        input_size: The size of the input tensor (height, width, channels).
        classes: The number of outputs the model should generate.

    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[0] or `input_size`[1] is not a multiple of 16.

    Returns:
        A TensorFlow ResNet9 model.
    """
    inp = layers.Input(shape=input_size)

    x = layers.Conv2D(32, 3, padding='same')(inp)
    x = GaussianBlur(sigma)(x)
    x = layers.MaxPool2D()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = GaussianBlur(sigma)(x)
    x = layers.MaxPool2D()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = GaussianBlur(sigma)(x)
    x = layers.MaxPool2D()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(classes)(x)
    x = layers.Activation('softmax', dtype='float32')(x)

    return tf.keras.Model(inputs=inp, outputs=x)


def CBSLeNet3(sigma: tf.Variable, input_size: Tuple[int, int, int] = (32, 32, 3), classes: int = 100) -> tf.keras.Model:
    """A small 9-layer ResNet Tensorflow model for cifar10 image classification.
    The model architecture is from https://github.com/davidcpage/cifar10-fast

    Args:
        input_size: The size of the input tensor (height, width, channels).
        classes: The number of outputs the model should generate.

    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[0] or `input_size`[1] is not a multiple of 16.

    Returns:
        A TensorFlow ResNet9 model.
    """
    inp = layers.Input(shape=input_size)

    x = layers.Conv2D(32, 3, padding='same')(inp)
    x = GaussianBlur(sigma)(x)
    x = layers.MaxPool2D()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = GaussianBlur(sigma)(x)
    x = layers.MaxPool2D()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = GaussianBlur(sigma)(x)
    x = layers.MaxPool2D()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, padding='same')(x)
    x = GaussianBlur(sigma)(x)
    x = layers.MaxPool2D()(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    # x = layers.Dense(256*2*2)(x)
    # x = layers.Dense(64, kernel_initializer='he_uniform')(x)
    x = layers.Dense(64)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(classes)(x)
    x = layers.Activation('softmax', dtype='float32')(x)

    return tf.keras.Model(inputs=inp, outputs=x)


def CBSLeNet(input_size: Tuple[int, int, int] = (32, 32, 3), classes: int = 100, n_models: int = 10) -> List[
    tf.keras.Model]:
    """A small 9-layer ResNet Tensorflow model for cifar10 image classification.
    The model architecture is from https://github.com/davidcpage/cifar10-fast

    Args:
        input_size: The size of the input tensor (height, width, channels).
        classes: The number of outputs the model should generate.

    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[0] or `input_size`[1] is not a multiple of 16.

    Returns:
        A TensorFlow ResNet9 model.
    """
    inps = [layers.Input(shape=input_size) for _ in range(n_models)]
    sigs = [1 * pow(0.9, p) for p in range(len(inps))]

    conv1 = layers.Conv2D(32, 3, padding='same')
    z = [conv1(x) for x in inps]
    # z = [gaussian_blur(x, sig) for x, sig in zip(z, sigs)]
    z = [gaussian_filter2d(x, sigma=sig, padding='CONSTANT') for x, sig in zip(z, sigs)]
    pool1 = layers.MaxPool2D()
    z = [pool1(x) for x in z]
    relu1 = layers.LeakyReLU(alpha=0.1)
    z = [relu1(x) for x in z]

    conv2 = layers.Conv2D(64, 3, padding='same')
    z = [conv2(x) for x in z]
    # z = [gaussian_blur(x, sig) for x, sig in zip(z, sigs)]
    z = [gaussian_filter2d(x, sigma=sig, padding='CONSTANT') for x, sig in zip(z, sigs)]
    pool2 = layers.MaxPool2D()
    z = [pool2(x) for x in z]
    relu2 = layers.LeakyReLU(alpha=0.1)
    z = [relu2(x) for x in z]

    conv3 = layers.Conv2D(64, 3, padding='same')
    z = [conv3(x) for x in z]
    # z = [gaussian_blur(x, sig) for x, sig in zip(z, sigs)]
    z = [gaussian_filter2d(x, sigma=sig, padding='CONSTANT') for x, sig in zip(z, sigs)]
    pool3 = layers.MaxPool2D()
    z = [pool3(x) for x in z]
    relu3 = layers.LeakyReLU(alpha=0.1)
    z = [relu3(x) for x in z]

    flat = layers.Flatten()
    z = [flat(x) for x in z]
    dense1 = layers.Dense(64)
    z = [dense1(x) for x in z]
    relu4 = layers.LeakyReLU(alpha=0.1)
    z = [relu4(x) for x in z]
    dense2 = layers.Dense(classes)
    z = [dense2(x) for x in z]
    sm = layers.Activation('softmax', dtype='float32')
    z = [sm(x) for x in z]

    models = [tf.keras.Model(inputs=inp, outputs=x) for inp, x in zip(inps, z)]

    return models


def gaussian_blur(img, kernel_size=3, sigma=1):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')

    # print(img.shape)
    # print(gaussian_kernel.shape)
    # gaussian_kernel = tf.tile(gaussian_kernel, [1, 1, 1, tf.shape(img)[-1]])
    # print(gaussian_kernel.shape)
    # # tf.print(gaussian_kernel)
    # # tf.print(gaussian_kernel.shape)
    # return tf.nn.conv2d(img, gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')

class Adjuster(Trace):
    def __init__(self, var: tf.Variable):
        super().__init__(mode='train')
        self.var = var

    def on_epoch_begin(self, data: Data) -> None:
        if self.system.epoch_idx % 5 == 0:
            self.var.assign(self.var * 0.9)
        data.write_with_log("sigma", self.var.numpy())


def get_estimator(epochs=100,
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
            RUA(inputs="x", outputs="x", mode="train"),
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            # PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            # RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            # Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            # CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
        ])

    # step 2
    # decay_frequency = 5
    # n_models = epochs // decay_frequency
    # models = fe.build(model_fn=lambda: CBSLeNet(n_models=n_models), optimizer_fn=["adam" for _ in range(n_models)])
    # network = fe.Network(ops=[
    #     EpochScheduler({i * decay_frequency + 1: ModelOp(model=model, inputs="x", outputs="y_pred") for i, model in
    #                     enumerate(models)}),
    #     CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    #     EpochScheduler({i * decay_frequency + 1: UpdateOp(model=model, loss_name="ce") for i, model in
    #                     enumerate(models)}),
    # ])
    sigma = tf.Variable(1.0, trainable=False)
    original_lr = 1e-1
    original_decay = 5e-4
    model = fe.build(model_fn=lambda: CBSLeNet3(sigma), optimizer_fn=lambda: tfa.optimizers.SGDW(weight_decay=original_decay, momentum=0.9, learning_rate=original_lr))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce"),
    ])

    # step 3
    traces = [
        MCC(true_key="y", pred_key="y_pred", output_name="mcc"),
        Accuracy(true_key="y", pred_key="y_pred", output_name="acc"),
        BestModelSaver(model=model, save_dir=save_dir, metric="mcc", save_best_mode="max", load_best_final=True),
        Adjuster(sigma),
        # LRScheduler(model=model, lr_fn="arc"),
        LRScheduler(model=model, lr_fn=lambda epoch: original_lr if epoch < 30 else original_lr/10 if epoch < 60 else original_lr/100 if epoch < 90 else original_lr/1000),
        DecayScheduler(model=model, lr_fn=lambda epoch: original_decay if epoch < 30 else original_decay/10 if epoch < 60 else original_decay/100 if epoch < 90 else original_decay/1000),
        # EarlyStopping(monitor="mcc", patience=20, compare='max'),
        Traceability('CBS_TF4')
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
    est.fit('cbs_rua_1')
    est.test()
