"""Tests for policy_utils."""

import os
from typing import List

from absl import flags
from absl.testing import absltest
import gin
from tf_agents.networks import actor_distribution_network

from compiler_opt.es import policy_utils
from compiler_opt.rl.regalloc import regalloc_network


def _get_abs_gin_files(filelist: List[str]) -> List[str]:
  return [os.path.join(flags.FLAGS.test_srcdir, f) for f in filelist]


class ConfigTest(absltest.TestCase):

  def test_inlining_config(self):
    gin.clear_config(clear_constants=True)
    gin.parse_config_files_and_bindings(
        config_files=_get_abs_gin_files([
            'google3/research/sir/compiler_opt/es/gin_configs/blackbox_learner.gin',
            'google3/research/sir/compiler_opt/es/gin_configs/inlining.gin'
        ]),
        bindings=[
            'llvm_size_path="bogus/llvm-size/path"', 'clang_path="bogus/clang"',
            'inlining.config.get_observation_processing_layer_creator.quantile_file_dir="{0}"'
            .format(
                os.path.join(
                    flags.FLAGS.test_srcdir,
                    'google3/third_party/ml_compiler_opt/compiler_opt/rl/inlining/vocab'
                ))
        ])
    actor_policy = policy_utils.create_actor_policy()
    self.assertIsNotNone(actor_policy)
    self.assertIsInstance(actor_policy._actor_network,
                          actor_distribution_network.ActorDistributionNetwork)

  def test_regalloc_config(self):
    gin.clear_config(clear_constants=True)
    gin.parse_config_files_and_bindings(
        config_files=_get_abs_gin_files([
            'google3/research/sir/compiler_opt/es/gin_configs/blackbox_learner.gin',
            'google3/research/sir/compiler_opt/es/gin_configs/regalloc.gin'
        ]),
        bindings=[
            'clang_path="bogus/clang"',
            'regalloc.config.get_observation_processing_layer_creator.quantile_file_dir="{0}"'
            .format(
                os.path.join(
                    flags.FLAGS.test_srcdir,
                    'google3/third_party/ml_compiler_opt/compiler_opt/rl/regalloc/vocab'
                ))
        ])
    actor_policy = policy_utils.create_actor_policy()
    self.assertIsNotNone(actor_policy)
    self.assertIsInstance(actor_policy._actor_network,
                          regalloc_network.RegAllocNetwork)


if __name__ == '__main__':
  absltest.main()