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

import importlib
from pathlib import Path

from eva.utils.metrics import Metrics

class Profiler:
    def __init__(self, filepath: str, classname: str):
        """
        * Profiler is supposed to be called post basic validation
        * So object creation is guaranteed
        * Create an object of classname
        """
        abs_path = Path(filepath).resolve()
        spec = importlib.util.spec_from_file_location(abs_path.stem, abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._classobj = getattr(module, classname)

    def run(self):
        """
        * Profiles the given UDF for various batch sizes
        * Gathers metrics related to time_taken and accuracy

        Param: None

        Returns: List[Metrics]
        """
        # TODO: Implement the actual logic
        # Use self._classobj's methods to run for various batch sizes

        return [Metrics(1, 25, 100),
                Metrics(2, 45, 100),
                Metrics(5, 110, 100)]
