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
"""Local ES trainer."""

# import concurrent.futures

from absl import logging
from compiler_opt.distributed.worker import Worker
from compiler_opt.distributed.local import local_worker_manager

import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin
from typing import List
import tensorflow as tf

from compiler_opt.es import blackbox_optimizers
from compiler_opt.es import gradient_ascent_optimization_algorithms
from compiler_opt.es import blackbox_learner_grpc
from compiler_opt.es import policy_utils
from compiler_opt.rl import policy_saver
from compiler_opt.rl import corpus

POLICY_NAME = "policy"

FLAGS = flags.FLAGS
_OUTPUT_PATH = flags.DEFINE_string("output_path", "",
                                   "Path to write all output")
_GIN_FILES = flags.DEFINE_multi_string(
    "gin_files", [], "List of paths to gin configuration files.")
_GIN_BINDINGS = flags.DEFINE_multi_string(
    "gin_bindings", [],
    "Gin bindings to override the values set in the config files.")
_TRAIN_CORPORA = flags.DEFINE_string("train_corpora", "",
                                           "List of paths to training corpora")
_TRAIN_CORPORA_WEIGHTS = flags.DEFINE_multi_float(
    "train_corpora_weights", [],
    "List of weights to assign to each corpus. Weights are normalized by the sum of the list to produce probabilities for each corpus during sampling."
)

_PRETRAINED_POLICY_PATH = flags.DEFINE_string(
    "pretrained_policy_path", None,
    "The path of the pretrained policy. If not provided, it will construct a new policy with randomly initialized weights."
)

_GREEDY = flags.DEFINE_bool(
    "greedy",
    None,
    "Whether to construct a greedy policy (argmax). If False, a sampling-based policy will be used.",
    required=True)

_REQUEST_DEADLINE = flags.DEFINE_float(
    "request_deadline", 30.0,
    "Deadline in seconds for requests to the data collection requests.")

_GRADIENT_ASCENT_OPTIMIZER_TYPE = flags.DEFINE_string(
    "gradient_ascent_optimizer_type", None,
    "Gradient ascent optimization algorithm: 'momentum' or 'adam'")
flags.mark_flag_as_required("gradient_ascent_optimizer_type")
_MOMENTUM = flags.DEFINE_float(
    "momentum", 0.0, "Momentum for momentum gradient ascent optimizer.")
_BETA1 = flags.DEFINE_float("beta1", 0.9,
                            "Beta1 for ADAM gradient ascent optimizer.")
_BETA2 = flags.DEFINE_float("beta2", 0.999,
                            "Beta2 for ADAM gradient ascent optimizer.")
_GRAD_REG_TYPE = flags.DEFINE_string(
    "grad_reg_type", "ridge",
    "Regularization method to use with regression gradient.")
_GRAD_REG_ALPHA = flags.DEFINE_float(
    "grad_reg_alpha", 0.01,
    "Weight of regularization term in regression gradient.")

@gin.configurable
def train(additional_compilation_flags=(), delete_compilation_flags=()):
  """Train with ES."""
  if not _TRAIN_CORPORA.value:
    raise ValueError("Need to supply nonempty train corpora.")

  # Get learner config
  learner_config = blackbox_learner_grpc.BlackboxLearnerConfig()

  # Create directories
  if not tf.io.gfile.isdir(_OUTPUT_PATH.value):
    tf.io.gfile.makedirs(_OUTPUT_PATH.value)

  # Construct the policy and upload it
  policy = policy_utils.create_actor_policy(greedy=_GREEDY.value)
  saver = policy_saver.PolicySaver({POLICY_NAME: policy})

  # Save the policy
  policy_save_path = os.path.join(_OUTPUT_PATH.value, "policy")
  saver.save(policy_save_path)

  # Get initial parameter
  if not _PRETRAINED_POLICY_PATH.value:
    # Use randomly initialized parameters
    logging.info("Use random parameters")
    initial_parameters = policy_utils.get_vectorized_parameters_from_policy(
        policy)
    logging.info("Parameter dimension: %s", initial_parameters.shape)
    logging.info("Initial parameters: %s", initial_parameters)
  else:
    # Read the parameters from the pretrained policy
    logging.info("Reading policy parameters from %s",
                 _PRETRAINED_POLICY_PATH.value)
    # Load the policy
    pretrained_policy = tf.saved_model.load(_PRETRAINED_POLICY_PATH.value)
    initial_parameters = policy_utils.get_vectorized_parameters_from_loaded_policy(
        pretrained_policy)

  policy_parameter_dimension = policy_utils.get_vectorized_parameters_from_policy(
      policy).shape[0]
  if policy_parameter_dimension != initial_parameters.shape[0]:
    raise ValueError("Pretrained policy dimension is incorrect")

  logging.info("Parameter dimension: %s", initial_parameters.shape)
  logging.info("Initial parameters: %s", initial_parameters)

  cps = corpus.create_corpus_for_testing(
      location=_TRAIN_CORPORA.value,
      elements=[corpus.ModuleSpec(name='smth', size=1)],
      additional_flags=additional_compilation_flags,
      delete_flags=delete_compilation_flags)
  
  # corp = corpus.Corpus(
  #   data_path=_TRAIN_CORPORA.value,
  #   additional_flags=additional_compilation_flags,
  #   delete_flags=delete_compilation_flags)

  # Construct policy saver
  saved_policy = policy_utils.create_actor_policy(greedy=True)
  policy_saver_function = functools.partial(
      policy_utils.save_policy,
      policy=saved_policy,
      save_folder=os.path.join(_OUTPUT_PATH.value, "saved_policies"))

  # the following are from Blackbox Library.
  init_current_input = initial_parameters
  init_iteration = 0
  metaparams = []  # Ignore meta params for state normalization for now
  # TODO(linzinan): delete all unused parameters.

  # ------------------ GRADIENT ASCENT OPTIMIZERS ------------------------------
  if _GRADIENT_ASCENT_OPTIMIZER_TYPE.value == "momentum":
    logging.info("Running momentum gradient ascent optimizer")
    # You can obtain a vanilla gradient ascent optimizer by setting momentum=0.0
    # and setting step_size to the desired learning rate.
    gradient_ascent_optimizer = gradient_ascent_optimization_algorithms.MomentumOptimizer(
        learner_config.step_size, _MOMENTUM.value)
  elif _GRADIENT_ASCENT_OPTIMIZER_TYPE.value == "adam":
    logging.info("Running Adam gradient ascent optimizer")
    gradient_ascent_optimizer = gradient_ascent_optimization_algorithms.AdamOptimizer(
        learner_config.step_size, _BETA1.value, _BETA2.value)
  else:
    logging.info("No gradient ascent optimizer selected. Stopping.")
    return
  # ----------------------------------------------------------------------------

  # ------------------ OPTIMIZERS ----------------------------------------------
  if learner_config.blackbox_optimizer == "MC_baseline":
    logging.info("Running ES/ARS. Filtering: %s directions",
                 str(learner_config.num_top_directions))
    blackbox_optimizer = blackbox_optimizers.MonteCarloBlackboxOptimizer(
        learner_config.precision_parameter, learner_config.est_type,
        learner_config.fvalues_normalization,
        learner_config.hyperparameters_update_method, metaparams, None,
        learner_config.num_top_directions, gradient_ascent_optimizer)
  elif learner_config.blackbox_optimizer == "trust_region":
    logging.info("Running trust region")
    tr_params = {
        "init_radius": FLAGS.tr_init_radius,
        "grow_threshold": FLAGS.tr_grow_threshold,
        "grow_factor": FLAGS.tr_grow_factor,
        "shrink_neg_threshold": FLAGS.tr_shrink_neg_threshold,
        "shrink_factor": FLAGS.tr_shrink_factor,
        "reject_threshold": FLAGS.tr_reject_threshold,
        "reject_factor": FLAGS.tr_reject_factor,
        "dense_hessian": FLAGS.tr_dense_hessian,
        "sub_termination": FLAGS.tr_sub_termination,
        "subproblem_maxiter": FLAGS.tr_subproblem_maxiter,
        "minimum_radius": FLAGS.tr_minimum_radius,
        "grad_type": FLAGS.grad_type,
        "grad_reg_type": _GRAD_REG_TYPE.value,
        "grad_reg_alpha": _GRAD_REG_ALPHA.value
    }
    for param in tr_params:
      logging.info("%s: %s", param, tr_params[param])
      blackbox_optimizer = blackbox_optimizers.TrustRegionOptimizer(
          learner_config.precision_parameter, learner_config.est_type,
          learner_config.fvalues_normalization,
          learner_config.hyperparameters_update_method, metaparams, tr_params)
  elif learner_config.blackbox_optimizer == "regression":
    logging.info("Running Regression Based Optimizer")
    blackbox_optimizer = blackbox_optimizers.SklearnRegressionBlackboxOptimizer(
        _GRAD_REG_TYPE.value, _GRAD_REG_ALPHA.value, learner_config.est_type,
        learner_config.fvalues_normalization,
        learner_config.hyperparameters_update_method, metaparams, None,
        gradient_ascent_optimizer)
  else:
    raise ValueError("Unknown optimizer: '{}'".format(
        learner_config.blackbox_optimizer))

  logging.info("Initializing blackbox learner.")
  learner = blackbox_learner_grpc.BlackboxLearner(
      blackbox_opt=blackbox_optimizer,
      sampler=cps,
      tf_policy_path=os.path.join(policy_save_path, POLICY_NAME),
      output_dir=_OUTPUT_PATH.value,
      policy_saver_fn=policy_saver_function,
      model_weights=init_current_input,
      config=learner_config,
      initial_step=init_iteration,
      deadline=_REQUEST_DEADLINE.value)

  logging.info("Ready to train: running for %d steps.",
               learner_config.total_steps)
  
  class TempWorker(Worker):
    """Temporary placeholder worker."""

    def __init__(self, arg, *, kwarg):
      self._token = 0
      self._arg = arg
      self._kwarg = kwarg

    def temp_compile(self, policy:List[bytes], samples:List[List[corpus.ModuleSpec]]):
      function_value = 1
      return function_value

  with local_worker_manager.LocalWorkerPoolManager(
      TempWorker, learner_config.total_num_perturbations, arg="", kwarg="") as pool:
    for _ in range(learner_config.total_steps):
      learner.run_step(pool)


def main(_):
  # Set gin files
  # gin_folder = "./compiler_opt/es/gin_configs"
  # gin.add_config_file_search_path(gin_folder)
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())
  
  train()


if __name__ == "__main__":
  app.run(main)
