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

    _min_accuracy = 0
    _max_deadline = float('inf')
    _favors = FavorType.ACCURACY

    @classmethod
    def print(self):
        print("acu=%d time=%f favour=%s" %(self._min_accuracy, self._max_deadline, self._favors))

    @classmethod
    def get_min_accuracy(self):
        return self._min_accuracy
    
    @classmethod
    def min_accuracy(self, value):
        self._min_accuracy = value
    
    @classmethod
    def get_max_deadline(self):
        return self._max_deadline
    
    @classmethod
    def max_deadline(self,value):
        self._max_deadline = value
    
    @classmethod
    def get_favors(self):
        return self._favors

    @classmethod
    def favors(self, favors):
        if (favors == 1):
            self._favors = FavorType.ACCURACY
        else:
            self._favors = FavorType.DEADLINE