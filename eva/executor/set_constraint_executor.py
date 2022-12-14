# coding=utf-8
# Copyright 2018-2022 EVA
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
# from typing import Iterator

from eva.executor.abstract_executor import AbstractExecutor
from eva.utils.optimizer_constraints import UDFOptimizerConstraints
from eva.planner.set_constraint_plan import SetConstraintPlan


class SetConstraintExecutor(AbstractExecutor):
    """
    Sets constraint to given value

    Arguments:
        node (AbstractPlan): The SetConstraint Plan

    """

    def __init__(self, node: SetConstraintPlan):
        super().__init__(node)
        # self._sample_freq = node.sample_freq.value
        self._min_accuracy = node.min_accuracy
        self._max_deadline = node.max_deadline
        self._favors = node.favors.value

    def validate(self):
        pass

    def exec(self):
        UDFOptimizerConstraints.min_accuracy(self._node.min_accuracy)
        UDFOptimizerConstraints.max_deadline(self._max_deadline)
        print("Setting favor = %s" %(self._favors))
        UDFOptimizerConstraints.favors(self._favors)
