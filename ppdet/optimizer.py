# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging

from paddle import fluid
import paddle.fluid.layers.ops as ops

import paddle.fluid.optimizer as optimizer
import paddle.fluid.regularizer as regularizer
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter

from ppdet.core.workspace import register, serializable

__all__ = ['LearningRate', 'OptimizerBuilder']

logger = logging.getLogger(__name__)


def cosine_decay_with_warmup(learning_rate,
                             max_iters=90000,
                             warmup_iters=1000,
                             warmup_factor=0.1):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    decrease lr for every mini-batch and start with warmup.
    """
    global_step = _decay_step_counter()
    lr = fluid.layers.tensor.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(global_step < warmup_iters):
            eta_min = learning_rate * warmup_factor
            decayed_lr = eta_min + (learning_rate - eta_min) * (global_step / warmup_iters)
            fluid.layers.tensor.assign(input=decayed_lr, output=lr)
        with switch.default():
            decayed_lr = learning_rate * \
                (ops.cos((global_step - warmup_iters) * (math.pi / (max_iters - warmup_iters))) + 1)/2
            fluid.layers.tensor.assign(input=decayed_lr, output=lr)
    return lr


@serializable
class CosineDecayWithWarmup(object):
    """
    Cosine learning rate decay with warmup

    Args:
        max_iters (int):
        warmup_iters (int): warm up steps
        warmup_factor (float): decay factor
    """

    def __init__(self, max_iters=None, warmup_iters=500, warmup_factor=0.1):
        super(CosineDecayWithWarmup, self).__init__()
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor

    def __call__(self, base_lr=None, learning_rate=None):
        assert base_lr is not None, "base LR should be provided"
        return cosine_decay_with_warmup(base_lr, self.max_iters, self.warmup_iters, self.warmup_factor)


@serializable
class PiecewiseDecay(object):
    """
    Multi step learning rate decay

    Args:
        gamma (float): decay factor
        milestones (list): steps at which to decay learning rate
    """

    def __init__(self, gamma=0.1, milestones=[60000, 80000], values=None):
        super(PiecewiseDecay, self).__init__()
        self.gamma = gamma
        self.milestones = milestones
        self.values = values

    def __call__(self, base_lr=None, learning_rate=None):
        if self.values is not None:
            return fluid.layers.piecewise_decay(self.milestones, self.values)
        assert base_lr is not None, "either base LR or values should be provided"
        values = [base_lr]
        lr = base_lr
        for _ in self.milestones:
            lr *= self.gamma
            values.append(lr)
        return fluid.layers.piecewise_decay(self.milestones, values)


@serializable
class LinearWarmup(object):
    """
    Warm up learning rate linearly

    Args:
        steps (int): warm up steps
        start_factor (float): initial learning rate factor
    """

    def __init__(self, steps=500, start_factor=1. / 3):
        super(LinearWarmup, self).__init__()
        self.steps = steps
        self.start_factor = start_factor

    def __call__(self, base_lr, learning_rate):
        start_lr = base_lr * self.start_factor

        return fluid.layers.linear_lr_warmup(
            learning_rate=learning_rate,
            warmup_steps=self.steps,
            start_lr=start_lr,
            end_lr=base_lr)


@register
class LearningRate(object):
    """
    Learning Rate configuration

    Args:
        base_lr (float): base learning rate
        schedulers (list): learning rate schedulers
    """
    __category__ = 'optim'

    def __init__(self,
                 base_lr=0.01,
                 schedulers=[PiecewiseDecay(), LinearWarmup()]):
        super(LearningRate, self).__init__()
        self.base_lr = base_lr
        self.schedulers = schedulers

    def __call__(self):
        lr = None
        for sched in self.schedulers:
            lr = sched(self.base_lr, lr)
        return lr


@register
class OptimizerBuilder():
    """
    Build optimizer handles

    Args:
        regularizer (object): an `Regularizer` instance
        optimizer (object): an `Optimizer` instance
    """
    __category__ = 'optim'

    def __init__(self,
                 regularizer={'type': 'L2',
                              'factor': .0001},
                 optimizer={'type': 'Momentum',
                            'momentum': .9}):
        self.regularizer = regularizer
        self.optimizer = optimizer

    def __call__(self, learning_rate):
        reg_type = self.regularizer['type'] + 'Decay'
        reg_factor = self.regularizer['factor']
        regularization = getattr(regularizer, reg_type)(reg_factor)

        optim_args = self.optimizer.copy()
        optim_type = optim_args['type']
        del optim_args['type']
        op = getattr(optimizer, optim_type)
        return op(learning_rate=learning_rate,
                  regularization=regularization,
                  **optim_args)
