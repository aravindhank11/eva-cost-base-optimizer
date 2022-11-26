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

class UDFOptimizerConstraints(object):
    """
    Optimizer constraints for selecting optimal UDFs
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(UDFOptimizerConstraints, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._min_accuracy = 0
        self._max_deadline = float('inf')
        self._favors = FavorType.ACCURACY

    @property
    def min_accuracy(self):
        return self._min_accuracy
    
    @min_accuracy.setter
    def min_accuracy(self,value):
        self._min_accuracy = value
    
    @property
    def max_deadline(self):
        return self._max_deadline
    
    @max_deadline.setter
    def max_deadline(self,value):
        self._max_deadline = value
    
    @property
    def favors(self):
        return self._favors

    @favors.setter
    def favors(self,value):
        self._favors = value
