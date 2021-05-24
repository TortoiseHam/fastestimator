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
from tensorflow.python.keras import layers
from tensorflow_addons.image import gaussian_filter2d

import fastestimator as fe
from fastestimator.dataset.data import cifair100
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler
from fastestimator.trace.io import BestModelSaver, Traceability
from fastestimator.trace.metric import MCC
from fastestimator.trace import Trace
from fastestimator.util import Data

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
    x = layers.Dense(256*2*2)(x)
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

class Adjuster(Trace):
    def __init__(self, var: tf.Variable):
        super().__init__(mode='train')
        self.var = var

    def on_epoch_begin(self, data: Data) -> None:
        if self.system.epoch_idx % 5 == 0:
            self.var.assign(self.var * 0.9)


def get_estimator(epochs=100,
                  batch_size=128,
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
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
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
    model = fe.build(model_fn=lambda: CBSLeNet3(sigma), optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce"),
    ])

    # step 3
    traces = [
        MCC(true_key="y", pred_key="y_pred", output_name="mcc"),
        BestModelSaver(model=model, save_dir=save_dir, metric="mcc", save_best_mode="max", load_best_final=True),
        Adjuster(sigma),
        Traceability('CBS_TF2')
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
    est.fit('cbs_1')
    est.test()
