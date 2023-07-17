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
"""Tests for ES trainer."""

import logging
import os
from absl import flags
from absl.testing import absltest
import gin
import numpy as np
from typing import List
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.es import blackbox_learner_grpc, policy_utils
from compiler_opt.rl import policy_saver

# def _get_abs_gin_files(filelist: List[str]) -> List[str]:
#   return [os.path.join(flags.FLAGS.test_srcdir, f) for f in filelist]

# POLICY_NAME = "policy"

# class GRPCTest(absltest.TestCase):

#   def test_get_policy_as_bytes(self):

#     # create a policy
#     gin.clear_config(clear_constants=True)
#     gin.parse_config_files_and_bindings(
#         config_files=_get_abs_gin_files([
#             'compiler_opt/es/gin_configs/blackbox_learner.gin',
#             'compiler_opt/es/gin_configs/regalloc.gin'
#         ]),
#         bindings=[
#             'clang_path="bogus/clang"',
#             'regalloc.config.get_observation_processing_layer_creator.quantile_file_dir="{0}"'
#             .format(
#                 os.path.join(
#                     flags.FLAGS.test_srcdir,
#                     'compiler_opt/rl/regalloc/vocab'
#                 ))
#         ])
#     policy = policy_utils.create_actor_policy()
#     saver = policy_saver.PolicySaver({POLICY_NAME: policy})

#     # Save the policy
#     testing_path = self.create_tempdir()
#     policy_save_path = os.path.join(testing_path, "temp_output/policy")
#     saver.save(policy_save_path)

#     self._tf_policy_path = os.path.join(policy_save_path, POLICY_NAME)
#     length_of_a_perturbation = 15353
#     params = np.arange(length_of_a_perturbation)

#     policy_as_bytes = blackbox_learner_grpc.BlackboxLearner._get_policy_as_bytes(self, params)
#     with open("./compiler_opt/es/get_policy_as_bytes_expected_output", 'rb') as f:
#       expected_output = f.read()
#     self.assertEqual(policy_as_bytes, expected_output)


# if __name__ == '__main__':
#   absltest.main()