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
from eva.expression.constant_value_expression import ConstantValueExpression
from eva.planner.abstract_plan import AbstractPlan
from eva.planner.types import PlanOprType


class SetConstraintPlan(AbstractPlan):
    """
    Arguments:
        min_accuracy: minimum accuracy constraint
        max_deadline: max deadline constraint
        favors: favors accuracy or deadline constraint
    """

    def __init__(self, min_accuracy, max_deadline, favors):
        self._min_accuracy = min_accuracy
        self._max_deadline = max_deadline
        self._favors = favors
        super().__init__(PlanOprType.SET_CONSTRAINT)

    @property
    def min_accuracy(self):
        return self._min_accuracy
    
    @property
    def max_deadline(self):
        return self._max_deadline
    
    @property
    def favors(self):
        return self._favors

    def __hash__(self) -> int:
        return hash((super().__hash__(),self._min_accuracy,self._max_deadline,self._favors))
