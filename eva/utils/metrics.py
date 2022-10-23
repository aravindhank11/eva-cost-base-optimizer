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


class Metrics:
    def __init__(self, batch_size: int, time_taken: int, accuracy: float):
        self._batch_size = batch_size
        self._time_taken = time_taken
        self._accuracy = accuracy

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def time_taken(self):
        return self._time_taken

    @property
    def accuracy(self):
        return self._accuracy
