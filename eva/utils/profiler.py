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
import time
from pathlib import Path

import torch
import importlib.util
import numpy as np
import pandas as pd

from eva.utils.metrics import Metrics


class Profiler:
    def __init__(self, filepath: str, classname: str, type_: str):
        """
        * Profiler is supposed to be called post basic validation
        * So object creation is guaranteed
        * Create an object of classname
        """
        abs_path = Path(filepath).resolve()
        spec = importlib.util.spec_from_file_location(abs_path.stem, abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._classobj = getattr(module, classname)()
        self.type = type_

    def run(self):
        """
        * Profiles the given UDF for various batch sizes
        * Gathers metrics related to time_taken

        Param: None

        Returns: List[Metrics]
        """

        #TODO: Remove the hack

        #expected_h = 960
        #expected_w = 540
        #expected_c = 3

        batch_sizes = [8]
        metrics_list=[]
        for batch in batch_sizes:
            inp = self._classobj.generate_sample_input(batch)
            """
            if (self.type == "ObjectDetection"):
                inp = torch.rand(batch, expected_c, expected_w, expected_h)
            else:
                inp = pd.DataFrame({"labels": [np.random.rand(batch)], "search": ["car"]})
            """
            start_time = time.time()
            _ = self._classobj.forward(inp)
            time_taken = time.time() - start_time
            metrics_obj = Metrics(batch, time_taken)
            metrics_list.append(metrics_obj)
        return metrics_list            
