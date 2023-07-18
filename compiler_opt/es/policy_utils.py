"""Util function to create a tf_agent policy."""

import gin
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from typing import Union
from tf_agents.policies import actor_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import tf_policy

from compiler_opt.rl import policy_saver
from compiler_opt.rl import registry


@gin.configurable(module='policy_utils')
def create_actor_policy(actor_network_ctor, greedy=False):
  """Creates an actor policy."""
  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()
  layers = tf.nest.map_structure(
      problem_config.get_preprocessing_layer_creator(),
      time_step_spec.observation)

  actor_network = actor_network_ctor(
      input_tensor_spec=time_step_spec.observation,
      output_tensor_spec=action_spec,
      preprocessing_layers=layers)

  policy = actor_policy.ActorPolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=actor_network)

  if greedy:
    policy = greedy_policy.GreedyPolicy(policy)

  return policy

def get_vectorized_parameters_from_policy(
    policy: Union[tf_policy.TFPolicy, tf.Module]):
  if isinstance(policy, tf_policy.TFPolicy):
    variables = policy.variables()
  elif policy.model_variables:
    variables = policy.model_variables

  parameters = [var.numpy().flatten() for var in variables]
  parameters = np.concatenate(parameters, axis=0)
  return parameters


def set_vectorized_parameters_for_policy(
    policy: Union[tf_policy.TFPolicy, tf.Module],
    parameters: npt.NDArray[np.float32]) -> None:
  if isinstance(policy, tf_policy.TFPolicy):
    variables = policy.variables()
  else:
    try:
      getattr(policy, 'model_variables')
    except AttributeError as e:
      raise TypeError(
        'policy must be a TFPolicy or a loaded SavedModel'
        ) from e
    variables = policy.model_variables

  param_pos = 0
  for variable in variables:
    shape = tf.shape(variable).numpy()
    num_ele = np.prod(shape)
    param = np.reshape(parameters[param_pos:param_pos + num_ele], shape)
    variable.assign(param)
    param_pos += num_ele
  if param_pos != len(parameters):
    raise ValueError(
        f'Parameter dimensions are not matched! Expected {len(parameters)} '
        'but only found {param_pos}.')


def save_policy(policy: tf_policy.TFPolicy,
                parameters,
                save_folder,
                policy_name):
  set_vectorized_parameters_for_policy(policy, parameters)
  saver = policy_saver.PolicySaver({policy_name: policy})
  saver.save(save_folder)
