"""Main routine of the RPC client coordinating RPC blackbox optimization."""

import dataclasses
import math
import multiprocessing.pool
import os
from typing import Any, Callable, List, Tuple

from absl import logging
import concurrent.futures
import gin
import grpc
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tf_agents  # Required for importing SM
from compiler_opt.distributed import buffered_scheduler

from compiler_opt.es import blackbox_optimizers, policy_utils
from compiler_opt.distributed.worker import Worker
from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.rl import data_collector, policy_saver, corpus

_DATA_THRESHOLDS = ((0.90, 0.0), (0.80, 0.6), (0.0, 1.0))

# If less than 40% of requests succeed, skip the step.
_SKIP_STEP_SUCCESS_RATIO = 0.4


@gin.configurable
@dataclasses.dataclass
class BlackboxLearnerConfig:
  """Hyperparameter configuration for BlackboxLearner."""

  # Total steps to train for
  total_steps: int

  # Name of the blackbox optimization algorithm
  blackbox_optimizer: str

  # What kind of ES training?
  #   - antithetic: for each perturbtation, try an antiperturbation
  #   - forward_fd: try total_num_perturbations independent perturbations
  est_type: blackbox_optimizers.EstimatorType

  # Should the rewards for blackbox optimization in a single step be normalized?
  fvalues_normalization: bool

  # How to update optimizer hyperparameters
  hyperparameters_update_method: blackbox_optimizers.UpdateMethod

  # Number of top performing perturbations to select in the optimizer
  # 0 means all
  num_top_directions: int

  # How many IR files to try a single perturbation on?
  num_ir_repeats_within_worker: int

  # How many times should we reuse IR to test different policies?
  num_ir_repeats_across_worker: int

  # How many IR files to sample from the test corpus at each iteration
  num_exact_evals: int

  # How many perturbations to attempt at each perturbation
  total_num_perturbations: int

  # How much to scale the stdev of the perturbations
  precision_parameter: float

  # Learning rate
  step_size: float


def _prune_skipped_perturbations(perturbations, rewards):
  """Remove perturbations that were skipped during the training step.

  Perturbations may be skipped due to an early exit condition or a server error
  (clang timeout, malformed training example, etc). The blackbox optimizer
  assumes that each perturbations has a valid reward, so we must remove any of
  these "skipped" perturbations.

  Pruning occurs in-place.

  Args:
    perturbations: the model perturbations used for the ES training step.
    rewards: the rewards for each perturbation.

  Returns:
    The number of perturbations that were pruned.
  """
  indices_to_prune = []
  for i, reward in enumerate(rewards):
    if reward is None:
      base = 2 * (i // 2)
      if base not in indices_to_prune:
        indices_to_prune.extend([base, base + 1])

  # Iterate in reverse so that the indices remain valid
  for i in reversed(indices_to_prune):
    del perturbations[i]
    del rewards[i]

  return len(indices_to_prune)


class Request:
  """Stores a future and its accociated tag"""
  def __init__(self, future: concurrent.futures.Future, samples: List[List[corpus.ModuleSpec]], tag: int):
    self.future = future
    self.samples = samples
    self.tag = tag
    self.function_value = None


def _handle_future(request: Request):
  """Returns the Request if the future has completed or None if it is not immediately ready.
  Logs an RpcError if there is one."""
  try:
    result = request.future.result(timeout=0.0)
    return request
  except grpc.FutureTimeoutError:
    return None
  except grpc.RpcError as err:
    logging.error('RPC error caught in collecting results: %s', str(err))
    return None


class BlackboxLearner(Worker):
  """Implementation of blackbox learning."""

  def __init__(self,
               blackbox_opt: blackbox_optimizers.BlackboxOptimizer,
               sampler: corpus.Corpus,
               tf_policy_path: str,
               output_dir: str,
               policy_saver_fn: Callable[..., Any],
               model_weights: npt.NDArray[np.float32],
               config: BlackboxLearnerConfig,
               initial_step: int = 0,
               deadline: float = 30.0):
    """Construct a BlackboxLeaner.

    Args:
      blackbox_opt: the blackbox optimizer to use
      train_sampler: corpus_sampler for training data.
      tf_policy_path: where to write the tf policy
      output_dir: the directory to write all outputs
      policy_saver_fn: function to save a policy to cns
      model_weights: the weights of the current model
      config: configuration for blackbox optimization.
      stubs: grpc stubs to inlining/regalloc servers
      initial_step: the initial step for learning.
      deadline: the deadline for requests to the inlining server.
    """
    self._blackbox_opt = blackbox_opt
    self._sampler = sampler
    self._tf_policy_path = tf_policy_path
    self._output_dir = output_dir
    self._policy_saver_fn = policy_saver_fn
    self._model_weights = model_weights
    self._config = config
    self._step = initial_step
    self._deadline = deadline

    # While we're waiting for the ES requests, we can
    # collect samples for the next round of training.
    self._samples = []

    self._summary_writer = tf.summary.create_file_writer(output_dir)

  def _get_perturbations(self) -> List[npt.NDArray[np.float32]]:
    """Get perturbations for the model weights."""
    perturbations = []
    for _ in range(self._config.total_num_perturbations):
      perturbations.append(
          np.random.normal(size=(len(self._model_weights))) *
          self._config.precision_parameter)
    return perturbations

  def _get_rewards(self, results: List[Request],
                   num_proposed_perturbations: int) -> List[float]:
    """Convert ES results to reward numbers."""
    rewards = [None] * num_proposed_perturbations

    def _get_index(tag):
      assert tag != 0
      if tag > 0:
        if self._config.est_type == 'antithetic':
          return (tag - 1) * 2
        else:
          return tag - 1
      if tag < 0:
        return (-tag - 1) * 2 + 1

    for result in results:
      index = _get_index(result.tag)
      if rewards[index] is None:
        rewards[index] = result.future.result()
      else:
        rewards[index] += result.future.result()

    return rewards

  def _update_model(self, perturbations: List[float], rewards: List[float]) -> None:
    """Update the model given a list of perturbations and rewards."""
    self._model_weights = self._blackbox_opt.run_step(
        perturbations=np.array(perturbations),
        function_values=np.array(rewards),
        current_input=self._model_weights,
        current_value=np.mean(rewards))

  def _log_rewards(self, rewards: List[float]):
    """Log reward to console."""
    logging.info('Train reward: [%f]', np.mean(rewards))

  def _log_tf_summary(self, rewards: List[float]) -> None:
    """Log tensorboard data."""
    with self._summary_writer.as_default():
      tf.summary.scalar(
          'reward/average_reward_train',
          np.mean(rewards),
          step=self._step)

      tf.summary.histogram(
          'reward/reward_train', rewards, step=self._step)

      train_regressions = [reward for reward in rewards if reward < 0]
      tf.summary.scalar(
          'reward/regression_probability_train',
          len(train_regressions) / len(rewards),
          step=self._step)

      tf.summary.scalar(
          'reward/regression_avg_train',
          np.mean(train_regressions),
          step=self._step)

      # The "max regression" is the min value, as the regressions are negative.
      tf.summary.scalar(
          'reward/regression_max_train',
          min(train_regressions, default=0),
          step=self._step)

      train_wins = [reward for reward in rewards if reward > 0]
      tf.summary.scalar(
          'reward/win_probability_train',
          len(train_wins) / len(rewards),
          step=self._step)

  def _save_model(self) -> None:
    """Save the model."""
    logging.info('Saving the model.')
    self._policy_saver_fn(
        parameters=self._model_weights,
        policy_name='iteration{}'.format(self._step))

  def _get_results(self, pool: local_worker_manager.LocalWorkerPoolManager, perturbations: List[bytes]) -> Tuple[List[float], List[float], List[float]]:
    if not self._samples:
      for _ in range(self._config.total_num_perturbations):
        self._samples.append(self._sampler.sample(
          self._config.num_ir_repeats_within_worker))

    # create tags
    tags = range(1, self._config.total_num_perturbations + 1)

    samples = self._samples
    # positive-negative pairs
    if self._config.est_type == 'antithetic':
      tags = [t for t in tags for t in (t, -t)]
      samples = [s for s in samples for s in (s, s)]

    compile_args = zip(perturbations, samples)
  
    _, futures = buffered_scheduler.schedule_on_worker_pool(
          action=lambda w, v: w.temp_compile(v[0],v[1]),
          jobs=compile_args,
          worker_pool=pool)
    
    # each perturbation has tag i and its negative (if antithetic) has tag -i
    requests: List[Request] = []
    for future, sample, tag in zip(futures, samples, tags):
      requests.append(Request(future, sample, tag))

    early_exit = data_collector.EarlyExitChecker(num_modules=len(futures),
                                                 deadline=self._deadline,
                                                 thresholds=_DATA_THRESHOLDS)

    # Collect next samples while we wait
    with multiprocessing.pool.ThreadPool(self._config.total_num_perturbations) as tpool:
      self._samples = []
      for _ in range(self._config.total_num_perturbations):
        self._samples.append(tpool.apply(self._sampler.sample, 
          [self._config.num_ir_repeats_within_worker]))
    
    # Wait for exit conditions
    early_exit.wait(lambda: sum([request.future.done() for request in requests]))

    # store the request if the future is done otherwise None
    raw_results = [_handle_future(request) for request in requests]
    # store only the completed requests
    done_results = [result for result in raw_results if result is not None]
    logging.info('[%d] requests out of [%d] did not terminate.',
                 len(raw_results) - len(done_results), len(raw_results))
    
    return raw_results, done_results

  def _get_policy_as_bytes(self, perturbation: npt.NDArray[np.float32]) -> List[bytes]:
    sm = tf.saved_model.load(self._tf_policy_path)
    # devectorize the perturbation
    policy_utils.set_vectorized_parameters_for_policy(sm, perturbation)

    sm_dir = '/tmp/sm'
    tf.saved_model.save(sm, sm_dir, signatures=sm.signatures)

    # convert to tflite
    # TODO(abenalaast): replace following with policy_saver.convert_mlgo_model(mlgo_model_dir: str, tflite_model_dir: str)
    tfl_dir = '/tmp/tfl'
    tf.io.gfile.makedirs(tfl_dir)
    tfl_path = os.path.join(tfl_dir, "model.tflite")
    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    tfl_model = converter.convert()
    with tf.io.gfile.GFile(tfl_path, "wb") as f:
        f.write(tfl_model)

    # copy json to tmp dir
    json_file = "output_spec.json"
    src_json = os.path.join(self._tf_policy_path, json_file)
    tf.io.gfile.copy(src_json, os.path.join(tfl_dir, json_file), True)

    # create and return policy
    policy_obj = policy_saver.Policy.from_filesystem(tfl_dir)
    return policy_obj.policy

  def run_step(self, pool: local_worker_manager.LocalWorkerPoolManager) -> None:
    """Run a single step of blackbox learning."""
    logging.info('-' * 80)
    logging.info('Step [%d]', self._step)

    initial_perturbations = self._get_perturbations()
    # positive-negative pairs
    if self._config.est_type == 'antithetic':
      initial_perturbations = [p for p in initial_perturbations for p in (p, -p)]

    # convert to bytes for compile job
    perturbations_as_bytes = []
    for perturbation in initial_perturbations:
      perturbations_as_bytes.append(self._get_policy_as_bytes(perturbation))

    raw_results, done_results = self._get_results(pool, perturbations_as_bytes)
    rewards = self._get_rewards(done_results, len(perturbations_as_bytes))

    num_pruned = _prune_skipped_perturbations(perturbations_as_bytes, rewards)
    logging.info('Pruned [%d]', num_pruned)
    min_num_rewards = math.ceil(_SKIP_STEP_SUCCESS_RATIO *
                                      len(raw_results))
    if (len(rewards) < min_num_rewards):
      logging.warning(
          'Skipping the step, too many requests failed: %d of %d '
          'train requests succeeded (required: %d)',
          len(rewards), len(raw_results), min_num_rewards)
      return

    self._update_model(initial_perturbations, rewards)
    self._log_rewards(rewards)
    self._log_tf_summary(rewards)

    self._save_model()

    self._step += 1
