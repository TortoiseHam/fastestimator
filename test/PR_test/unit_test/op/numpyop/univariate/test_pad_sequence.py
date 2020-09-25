# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
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
import unittest

import numpy as np

from fastestimator.op.numpyop.univariate import PadSequence
from fastestimator.test.unittest_util import is_equal


class TestPadSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.array([1, 2, 3, 4])]
        cls.single_output = [np.array([1, 2, 3, 4, 0, 0, 0])]
        cls.multi_input = [np.array([2, 2]), np.array([0, 1, 2])]
        cls.multi_output = [np.array([2, 2, -1, -1]), np.array([0, 1, 2, -1])]

    def test_single_input(self):
        op = PadSequence(inputs='x', outputs='x', max_len=7, value=0)
        data = op.forward(data=self.single_input, state={})
        self.assertTrue(is_equal(data, self.single_output))

    def test_multi_input(self):
        op = PadSequence(inputs='x', outputs='x', max_len=4, value=-1)
        data = op.forward(data=self.multi_input, state={})
        self.assertTrue(is_equal(data, self.multi_output))

    def test_sequence_truncate(self):
        op = PadSequence(inputs='x', outputs='x', max_len=3)
        data = op.forward(data=self.single_input, state={})
        self.assertTrue(is_equal(data, [np.array([1, 2, 3])]))

    def test_sequene_prepent(self):
        op = PadSequence(inputs='x', outputs='x', max_len=7, value=0, append=False)
        data = op.forward(data=self.single_input, state={})
        self.assertTrue(is_equal(data, [np.array([0, 0, 0, 1, 2, 3, 4])]))
