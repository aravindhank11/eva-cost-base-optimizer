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
from eva.parser.statement import AbstractStatement
from eva.parser.types import StatementType


class SetConstraintStatement(AbstractStatement):
    """Set Constraint Statement constructed after parsing the input query

    Attributes:
        min_accuracy: minimum accuracy constraint
        max_deadline: max deadline constraint
        favors: favors accuracy or deadline constraint
    """

    def __init__(self, min_accuracy, max_deadline, favors):
        super().__init__(StatementType.SET_CONSTRAINT)
        self._min_accuracy = min_accuracy
        self._max_deadline = max_deadline
        self._favors = favors

    @property
    def min_accuracy(self):
        return self._min_accuracy
    
    @property
    def max_deadline(self):
        return self._max_deadline
    
    @property
    def favors(self):
        return self._favors

    def __str__(self) -> str:
        return "SET CONSTRAINT MIN_ACCURACY {} MAX_DEADLINE {} FAVORS {}".format(self._min_accuracy,self._max_deadline,self._favors)

    def __eq__(self, other):
        if not isinstance(other, SetConstraintStatement):
            return False
        return self.min_accuracy == other.min_accuracy and self.max_deadline == other.max_deadline and self.favors == other.favors

    def __hash__(self) -> int:
        return hash((super().__hash__(),self.min_accuracy,self.max_deadline,self.favors))
