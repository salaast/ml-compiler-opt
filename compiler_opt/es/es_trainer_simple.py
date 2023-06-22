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

from absl import app
import concurrent.futures
import numpy as np

from absl import logging
from compiler_opt.distributed.worker import Worker
from compiler_opt.distributed import buffered_scheduler
from compiler_opt.distributed.local import local_worker_manager

class JobStep(Worker):
  """Test worker."""

  def __init__(self, arg, *, kwarg):
    self._token = 0
    self._arg = arg
    self._kwarg = kwarg

  @classmethod
  def is_priority_method(cls, method_name: str) -> bool:
    return method_name == 'priority_method'

  def priority_method(self):
    return f'priority {self._token}'

  def get_token(self):
    return self._token

  def set_token(self, value):
    self._token = value

  def get_arg(self):
    return self._arg

  def get_kwarg(self):
    return self._kwarg

  def run_step(self, module, policy=1):
    return module * module + policy

  def run_both_pertubations(self, module, p1=1, p2=1):
    return (module * module * p1, module * module * p2)


def train_simple(
    modules=range(3), num_workers=3, num_iter=1, arg=1, kwarg=2, policy=1):
  logging.info("initial policy: "+str(policy))
  with local_worker_manager.LocalWorkerPoolManager(
      JobStep, num_workers, arg, kwarg=kwarg) as pool:
    for _ in range(num_iter):
      # pertubations of policy
      p1 = policy
      p2 = -policy
      # for each module, assign work to workers and store the result in a
      # list called futures. The workers have not finished the task so this
      # is a list of futures. Call run_step with arguments formatted as
      # [(module1, p1), (module1, p2), (module2, p1), ...] and return as
      # [m1p1 future, m1p2 future, m2p1 future, ...]
      workers, futures = buffered_scheduler.schedule_on_worker_pool(
          action=lambda w, v: w.run_step(v[0], v[1]),
          jobs=[(module, p) for module in modules for p in (p1, p2)],
          worker_pool=pool)

      # while some work is not done
      not_done = futures
      logging.info("start while")
      results = []
      while not_done:
        # update lists as work gets done
        done, not_done = concurrent.futures.wait(
            not_done, return_when="FIRST_COMPLETED")
        # do something with done futures
        for future in done:
          results.append(future.result())
      # once all work is done, a new policy will be formed
      logging.info("results: "+" ".join(str(result) for result in results))
      gradient = np.sqrt(np.average(results))
      logging.info("gradient: "+str(gradient))
      policy += gradient
      logging.info("policy: "+str(policy))
    return policy

def main(_):
  train_simple()


if __name__ == "__main__":
  app.run(main)