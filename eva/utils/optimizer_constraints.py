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

from enum import Enum, auto

class FavorType(Enum):
    ACCURACY = auto()
    DEADLINE = auto()

class UDFOptimizerConstraints:
    """
    Optimizer constraints for selecting optimal UDFs
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(UDFOptimizerConstraints, cls).__new__(cls)
        return cls.instance

    def __init__(self,min_accuracy=0,max_deadline=float('inf'),favors=FavorType.ACCURACY):
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
