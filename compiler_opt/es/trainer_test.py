# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for local ES trainer."""


from absl.testing import absltest
from tf_agents.system import system_multiprocessing as multiprocessing
from compiler_opt.es import trainer


class TrainerTest(absltest.TestCase):
    def test_trainer(self):
        # modules = range(3)
        # num_workers = 2
        # num_iter = 1
        # arg = 1
        # kwarg = 2
        # policy = 1


        trainer.train()


if __name__ == '__main__':
    multiprocessing.handle_test_main(absltest.main)
